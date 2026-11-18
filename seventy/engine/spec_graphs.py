"""CUDA graphs of whole speculative decoding steps, one per batch bucket B.

One replay runs: the target's verify step over [anchor, d1 .. d7] for B sequences, acceptance on the device
(seventy.engine.spec.device_accept, cut at stop tokens and token limits), the Gated DeltaNet commit of the
accepted rows, the drafter context of all
eight rows (rows past the accepted ones are overwritten by the next block before anything reads them), and the
next draft from the new anchor. The host reads back the accepted counts, new tokens and next drafts once.

Every rank replays the same graph on the same inputs, which come from the host (including the random numbers
drawn from each request's generator), so all ranks compute the same acceptance without extra communication.
Rows beyond the real batch point at the scratch block and slot and are greedy.
"""

from __future__ import annotations

import torch

from .cache import ModelCache
from .spec import KMAX, STOP_SLOTS, cut_at_stops, device_accept
from .step import StepInput


class SpecGraphs:
    def __init__(self, model, drafter, cache: ModelCache, device: torch.device, max_blocks_per_seq: int,
                 buckets: tuple[int, ...]):
        self.model, self.drafter, self.cache, self.device = model, drafter, cache, device
        self.width = max_blocks_per_seq
        self.block = drafter.cfg.block
        self.top = drafter.cfg.top_k
        self.layers = drafter.cfg.target_layers
        self.buckets = tuple(sorted(buckets))
        self.graphs: dict[int, tuple] = {}

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def _inputs(self, b: int) -> dict[str, torch.Tensor]:
        dev, k, c = self.device, self.block - 1, self.top
        return {
            "tokens": torch.zeros(b, self.block, dtype=torch.long, device=dev),
            "starts": torch.zeros(b, dtype=torch.long, device=dev),
            "delta": torch.zeros(b, dtype=torch.long, device=dev),
            "slots": torch.full((b,), self.cache.scratch_slot, dtype=torch.long, device=dev),
            "tables": torch.full((b, self.width), self.cache.scratch_block, dtype=torch.int32, device=dev),
            "temp": torch.zeros(b, device=dev),
            "top_k": torch.ones(b, dtype=torch.long, device=dev),
            "top_p": torch.ones(b, device=dev),
            "min_p": torch.zeros(b, device=dev),
            "q": torch.zeros(b, k, c, device=dev),
            "cand": torch.zeros(b, k, c, dtype=torch.long, device=dev),
            "u_accept": torch.zeros(b, k, device=dev),
            "u_residual": torch.zeros(b, k + 1, device=dev),
            "gumbel": torch.zeros(b, k, c, device=dev),
            "stop": torch.full((b, STOP_SLOTS), -1, dtype=torch.long, device=dev),
            "limit": torch.full((b,), self.block, dtype=torch.long, device=dev),
        }

    def _body(self, b: int, x: dict[str, torch.Tensor]):
        blk, bs, dev = self.block, self.cache.block_size, self.device
        offs = torch.arange(blk, device=dev)
        pos = x["starts"][:, None] + offs
        kv = x["tables"].gather(1, (pos // bs).int().long()).long() * bs + pos % bs
        step = StepInput(
            token_ids=x["tokens"].view(-1),
            positions=(pos + x["delta"][:, None]).reshape(1, -1).expand(3, -1).contiguous(),
            block_tables=x["tables"], logit_rows=torch.arange(b * blk, device=dev),
            query_start=[i * blk for i in range(b + 1)], context_lens=[0] * b, state_slots=[0] * b, block_size=bs,
            query_start_t=torch.arange(b + 1, dtype=torch.int32, device=dev) * blk,
            context_lens_t=x["starts"].int(), state_slots_t=x["slots"].int(), kv_slots_t=kv.view(-1).int(),
            max_blocks=0,
        )
        stash: dict = {}
        logits, cap = self.model.forward(step, self.cache, capture=self.layers, stash=stash)
        n, new = device_accept(logits.view(b, blk, -1), x["tokens"][:, 1:], x["q"], x["cand"], x["temp"],
                               x["top_k"], x["top_p"], x["min_p"], x["u_accept"], x["u_residual"])
        n = cut_at_stops(n, new, x["stop"], x["limit"])
        self.model.commit(step, self.cache, stash, (n + 1).int())
        feats = torch.cat([cap[i] for i in self.layers], dim=-1)
        self.drafter.write_context(feats, pos.reshape(-1), x["slots"].repeat_interleave(blk))
        rows = torch.arange(b, device=dev)
        anchors = new[rows, n]
        h = self.drafter.hidden(anchors, x["starts"] + n + 1, x["slots"]).view(b, blk, -1)[:, 1:]
        h = h.reshape(b * (blk - 1), -1)
        values, ids = self.drafter._candidates(h)
        d = self.drafter.walk(anchors, values.view(b, blk - 1, -1), ids.view(b, blk - 1, -1), h, x["temp"],
                              x["gumbel"])
        return {"n": n, "new": new, "drafts": d.tokens, "probs": d.probs, "cand": d.candidates}

    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        for b in sorted(self.buckets, reverse=True):
            x = self._inputs(b)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._body(b, x)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self._body(b, x)
            self.graphs[b] = (graph, x, out)
        torch.cuda.synchronize(self.device)

    def run(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        """rows: per sequence {tokens [block], start, delta (rotary position minus token index), slot, table, temp,
        top_k, top_p, min_p, q, cand, u_accept, u_residual, gumbel, stop (up to STOP_SLOTS ids), limit (most new
        tokens)} (host values). Returns the bucket's output tensors (valid for the first len(rows))."""
        b = self.bucket(len(rows))
        graph, x, out = self.graphs[b]
        n = len(rows)
        host = {k: torch.zeros_like(v, device="cpu") for k, v in x.items() if k != "tables"}
        host["slots"].fill_(self.cache.scratch_slot)
        host["top_k"].fill_(1)
        host["top_p"].fill_(1.0)
        host["stop"].fill_(-1)
        host["limit"].fill_(self.block)
        width = max(len(r["table"]) for r in rows)
        table = torch.full((b, width), self.cache.scratch_block, dtype=torch.int32)
        for i, r in enumerate(rows):
            host["tokens"][i] = torch.as_tensor(r["tokens"])
            host["starts"][i] = r["start"]
            host["delta"][i] = r.get("delta", 0)
            host["slots"][i] = r["slot"]
            table[i, : len(r["table"])] = torch.as_tensor(r["table"], dtype=torch.int32)
            for key in ("temp", "top_k", "top_p", "min_p"):
                host[key][i] = r[key]
            stops = r.get("stop", ())
            host["stop"][i, : len(stops)] = torch.as_tensor(list(stops), dtype=torch.long)
            host["limit"][i] = r.get("limit", self.block)
            for key in ("q", "cand", "u_accept", "u_residual", "gumbel"):
                if r.get(key) is not None:
                    host[key][i] = torch.as_tensor(r[key])
        for k, v in host.items():
            x[k].copy_(v, non_blocking=True)
        x["tables"][:, :width].copy_(table, non_blocking=True)
        graph.replay()
        return {k: v[:n] for k, v in out.items()}


def eligible(params, top_k_max: int = KMAX) -> bool:
    """Whether a request's sampling fits the graph's device acceptance."""
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return False
    if params.repetition_penalty != 1.0 or params.presence_penalty or params.frequency_penalty:
        return False
    if params.temperature == 0:
        return True
    return 1 <= params.top_k <= top_k_max
