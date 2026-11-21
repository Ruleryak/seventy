"""CUDA graphs of whole speculative decoding steps, one per batch bucket B.

One replay runs: the target's verify step over [anchor, d1 .. d7] for B sequences, acceptance on the device
(seventy.engine.spec.device_accept_top on each rank's vocabulary top k, cut at stop tokens and token limits), the Gated DeltaNet commit of the
accepted rows, the drafter context of all
eight rows (rows past the accepted ones are overwritten by the next block before anything reads them), and the
next draft from the new anchor. The host reads back the accepted counts, new tokens and next drafts once.

Every rank replays the same graph on the same inputs, which come from the host (including the random numbers
drawn from each request's generator), so all ranks compute the same acceptance without extra communication.
Rows beyond the real batch point at the scratch block and slot and are greedy.

Inputs: all but the block tables live in one device buffer per bucket (the graph reads views of it). Rank 0 fills the
matching pinned host buffer through numpy views and sends its bytes to the other ranks; every rank copies them with
one transfer. The block tables are copied up to the widest row.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .cache import ModelCache
from .spec import KMAX, STOP_SLOTS, cut_at_stops, device_accept_top
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
        self.packed: dict[int, tuple] = {}
        self.sent: dict[int, tuple] = {}
        self.slot_tables: dict[int, np.ndarray] = {}
        self.slot_len: dict[int, int] = {}

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def _fields(self, b: int) -> list[tuple[str, tuple[int, ...], torch.dtype, float]]:
        """(name, shape, dtype, padding value) of the packed inputs of bucket b."""
        k, c = self.block - 1, self.top
        return [
            ("tokens", (b, self.block), torch.long, 0), ("starts", (b,), torch.long, 0), ("delta", (b,), torch.long, 0),
            ("slots", (b,), torch.long, self.cache.scratch_slot), ("temp", (b,), torch.float32, 0.0),
            ("top_k", (b,), torch.long, 1), ("top_p", (b,), torch.float32, 1.0), ("min_p", (b,), torch.float32, 0.0),
            ("q", (b, k, c), torch.float32, 0.0), ("cand", (b, k, c), torch.long, 0),
            ("u_accept", (b, k), torch.float32, 0.0), ("u_residual", (b, k + 1), torch.float32, 0.0),
            ("gumbel", (b, k, c), torch.float32, 0.0), ("stop", (b, STOP_SLOTS), torch.long, -1),
            ("limit", (b,), torch.long, self.block),
        ]

    def _inputs(self, b: int) -> dict[str, torch.Tensor]:
        fields = self._fields(b)
        offsets, total = [], 0
        for _, shape, dtype, _ in fields:
            nbytes = int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
            offsets.append(total)
            total += -(-nbytes // 16) * 16
        dev_buf = torch.zeros(total, dtype=torch.uint8, device=self.device)
        host_buf = torch.zeros(total, dtype=torch.uint8).pin_memory()
        x, views, defaults = {}, {}, host_buf.numpy().copy()
        for (name, shape, dtype, pad), off in zip(fields, offsets):
            nbytes = int(np.prod(shape)) * torch.tensor([], dtype=dtype).element_size()
            x[name] = dev_buf[off : off + nbytes].view(dtype).view(shape)
            views[name] = host_buf[off : off + nbytes].view(dtype).view(shape).numpy()
            defaults[off : off + nbytes].view(views[name].dtype)[:] = pad
        x["tables"] = torch.full((b, self.width), self.cache.scratch_block, dtype=torch.int32, device=self.device)
        self.packed[b] = (dev_buf, host_buf, views, defaults)
        dev_buf.copy_(torch.from_numpy(defaults))
        return x

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
        (pred, vals, idx), cap = self.model.forward(step, self.cache, capture=self.layers, stash=stash, top=KMAX)
        if vals.shape[-1] == KMAX and os.environ.get("SEVENTY_SPEC_ACCEPT") != "torch":
            from ..ops import kernels

            n, new = kernels.spec_accept(pred.view(b, blk), vals.view(b, blk, -1), idx.view(b, blk, -1),
                                         x["tokens"][:, 1:], x["q"], x["cand"], x["temp"], x["top_k"], x["top_p"],
                                         x["min_p"], x["u_accept"], x["u_residual"], x["stop"], x["limit"])
        else:
            n, new = device_accept_top(pred.view(b, blk), vals.view(b, blk, -1), idx.view(b, blk, -1),
                                       x["tokens"][:, 1:], x["q"], x["cand"], x["temp"], x["top_k"], x["top_p"],
                                       x["min_p"], x["u_accept"], x["u_residual"])
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

    def pack(self, rows: list[dict]) -> tuple:
        """rows: per sequence {tokens [block], start, delta (rotary position minus token index), slot, table, temp,
        top_k, top_p, min_p, q, cand, u_accept, u_residual, gumbel, stop (up to STOP_SLOTS ids), limit (most new
        tokens)} (host values), and table_key (sequence id, blocks epoch) to send only appended blocks. Returns
        what run_packed takes (and what rank 0 sends to the other ranks): the block tables travel as updates of
        each rank's per-slot tables, whole when a slot's sequence or block list changed, else the appended blocks."""
        b = self.bucket(len(rows))
        _, host_buf, views, defaults = self.packed[b]
        flat = host_buf.numpy()
        flat[:] = defaults
        for i, r in enumerate(rows):
            views["tokens"][i] = r["tokens"]
            views["starts"][i] = r["start"]
            views["delta"][i] = r.get("delta", 0)
            views["slots"][i] = r["slot"]
            views["temp"][i] = r["temp"]
            views["top_k"][i] = r["top_k"]
            views["top_p"][i] = r["top_p"]
            views["min_p"][i] = r["min_p"]
            stops = r.get("stop", ())
            views["stop"][i, : len(stops)] = stops
            views["limit"][i] = r.get("limit", self.block)
            for key in ("q", "cand", "u_accept", "u_residual", "gumbel"):
                v = r.get(key)
                if v is not None:
                    views[key][i] = v.numpy() if isinstance(v, torch.Tensor) else v
        return b, len(rows), flat.tobytes(), self._table_updates(rows)

    def _table_updates(self, rows: list[dict]) -> list[tuple]:
        """(slot, first, total, ids) for each row whose block table differs from what the ranks have: the appended
        blocks when the slot holds the same sequence with the same block list (table_key), else the whole table."""
        updates = []
        for r in rows:
            slot, blocks, key = r["slot"], r["table"], r.get("table_key")
            total = len(blocks)
            sent = self.sent.get(slot)
            same = key is not None and sent is not None and sent[0] == key
            if same and sent[1] == total:
                continue
            first = sent[1] if same and sent[1] < total else 0
            updates.append((slot, first, total, np.asarray(blocks[first:], dtype=np.int32)))
            self.sent[slot] = (key, total)
        return updates

    def _apply_tables(self, updates: list) -> None:
        """Bring this rank's per-slot block tables up to date: (slot, first, total, ids) sets entries
        [first, total) of the slot's table to ids."""
        for slot, first, total, ids in updates:
            t = self.slot_tables.get(slot)
            if t is None or len(t) < total:
                grown = np.full(max(total, 64, 2 * (len(t) if t is not None else 0)), self.cache.scratch_block,
                                dtype=np.int32)
                if t is not None:
                    grown[:first] = t[:first]
                t = self.slot_tables[slot] = grown
            t[first:total] = ids
            self.slot_len[slot] = total

    def run_packed(self, b: int, n: int, data: bytes, updates: list) -> dict[str, torch.Tensor]:
        graph, x, out = self.graphs[b]
        dev_buf, host_buf, views, _ = self.packed[b]
        host_buf.numpy()[:] = np.frombuffer(data, dtype=np.uint8)
        dev_buf.copy_(host_buf, non_blocking=True)
        self._apply_tables(updates)
        slots = [int(s) for s in views["slots"][:n]]
        width = max(self.slot_len[s] for s in slots)
        table = np.full((b, width), self.cache.scratch_block, dtype=np.int32)
        for i, s in enumerate(slots):
            table[i, : self.slot_len[s]] = self.slot_tables[s][: self.slot_len[s]]
        x["tables"][:, :width].copy_(torch.from_numpy(table), non_blocking=True)
        graph.replay()
        return {k: v[:n] for k, v in out.items()}

    def run(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        return self.run_packed(*self.pack(rows))


def eligible(params, top_k_max: int = KMAX) -> bool:
    """Whether a request's sampling fits the graph's device acceptance."""
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return False
    if params.repetition_penalty != 1.0 or params.presence_penalty or params.frequency_penalty:
        return False
    if params.temperature == 0:
        return True
    return 1 <= params.top_k <= top_k_max
