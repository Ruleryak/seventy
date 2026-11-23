"""CUDA graphs of whole speculative decoding steps, one per batch bucket B.

One replay runs: the target's verify step over [anchor, d1 .. d(V-1)] for B sequences (V rows: the whole drafter
block, or fewer when B is large, see verify_rows), acceptance on the device (seventy.engine.spec.device_accept_top on
each rank's vocabulary top k, cut at stop tokens and token limits), the Gated DeltaNet commit of the accepted rows,
the drafter context of all V rows (rows past the accepted ones are overwritten by the next block before anything
reads them), and the next draft (a whole block) from the new anchor. The host reads back the accepted counts, new
tokens and next drafts once.

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
from ..model.qwen3_5 import FINAL
from .spec import KMAX, STOP_SLOTS, cut_at_stops, device_accept_top
from .step import StepInput


ROW_BUDGET = 32
MIN_ROWS = 4


def verify_rows(batch: int, block: int, budget: int | None = None) -> int:
    """Rows that a speculative step verifies for each of batch sequences: the anchor and the first drafts of the
    drafter's block. Up to budget rows in total (default ROW_BUDGET, SEVENTY_SPEC_ROW_BUDGET; 0: always the whole
    block), each sequence gets the whole block; above it, budget // batch rows, but at least MIN_ROWS.

    The verify GEMMs of a step cost about as much as the rows they compute, while the drafts late in a block are
    seldom accepted. Greedy probes of DFlash2 (block 8) on 4 V100s, 2K to 100K tokens of context, gave the most
    tokens per second at 8 rows for 1 and 2 streams, about the same from 5 to 8 rows for 4 streams, 5 rows for 6
    streams (+13 % to +19 %) and 4 rows for 8 streams (+15 % to +22 %): about 32 rows per step."""
    if budget is None:
        budget = int(os.environ.get("SEVENTY_SPEC_ROW_BUDGET", ROW_BUDGET))
    if budget <= 0:
        return block
    return min(block, max(MIN_ROWS, budget // batch))


class SpecGraphs:
    def __init__(self, model, drafter, cache: ModelCache, device: torch.device, max_blocks_per_seq: int,
                 buckets: tuple[int, ...]):
        self.model, self.drafter, self.cache, self.device = model, drafter, cache, device
        self.width = max_blocks_per_seq
        self.block = drafter.cfg.block
        self.top = drafter.cfg.top_k
        self.layers = drafter.cfg.target_layers
        self.mtp = hasattr(drafter, "after_verify")
        self.buckets = tuple(sorted(buckets))
        self.rows = {b: self.block if self.mtp else verify_rows(b, self.block) for b in self.buckets}
        self.graphs: dict[int, tuple] = {}
        self.packed: dict[int, tuple] = {}
        self.sent: dict[int, tuple] = {}
        self.slot_tables: dict[int, np.ndarray] = {}
        self.slot_len: dict[int, int] = {}

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def _fields(self, b: int) -> list[tuple[str, tuple[int, ...], torch.dtype, float]]:
        """(name, shape, dtype, padding value) of the packed inputs of bucket b."""
        v, c = self.rows[b], self.top
        k = v - 1
        return [
            ("tokens", (b, v), torch.long, 0), ("starts", (b,), torch.long, 0), ("delta", (b,), torch.long, 0),
            ("slots", (b,), torch.long, self.cache.scratch_slot), ("temp", (b,), torch.float32, 0.0),
            ("top_k", (b,), torch.long, 1), ("top_p", (b,), torch.float32, 1.0), ("min_p", (b,), torch.float32, 0.0),
            ("q", (b, k, c), torch.float32, 0.0), ("cand", (b, k, c), torch.long, 0),
            ("u_accept", (b, k), torch.float32, 0.0), ("u_residual", (b, k + 1), torch.float32, 0.0),
            ("gumbel", (b, self.block - 1, c), torch.float32, 0.0), ("stop", (b, STOP_SLOTS), torch.long, -1),
            ("limit", (b,), torch.long, v),
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
        blk, bs, dev = self.rows[b], self.cache.block_size, self.device
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
        if self.mtp:
            d = self.drafter.after_verify(step, cap[FINAL], new, n, x["starts"], x["tables"], x["delta"], x["slots"],
                                          self.cache)
            return {"n": n, "new": new, "drafts": d.tokens, "probs": d.probs, "cand": d.candidates}
        feats = torch.cat([cap[i] for i in self.layers], dim=-1)
        self.drafter.write_context(feats, pos.reshape(-1), x["slots"].repeat_interleave(blk))
        rows = torch.arange(b, device=dev)
        anchors = new[rows, n]
        db = self.block
        h = self.drafter.hidden(anchors, x["starts"] + n + 1, x["slots"]).view(b, db, -1)[:, 1:]
        h = h.reshape(b * (db - 1), -1)
        values, ids = self.drafter._candidates(h)
        d = self.drafter.walk(anchors, values.view(b, db - 1, -1), ids.view(b, db - 1, -1), h, x["temp"],
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
        v = self.rows[b]
        for i, r in enumerate(rows):
            views["tokens"][i] = r["tokens"][:v]
            views["starts"][i] = r["start"]
            views["delta"][i] = r.get("delta", 0)
            views["slots"][i] = r["slot"]
            views["temp"][i] = r["temp"]
            views["top_k"][i] = r["top_k"]
            views["top_p"][i] = r["top_p"]
            views["min_p"][i] = r["min_p"]
            stops = r.get("stop", ())
            views["stop"][i, : len(stops)] = stops
            views["limit"][i] = min(r.get("limit", v), v)
            for key in ("q", "cand", "u_accept", "u_residual", "gumbel"):
                val = r.get(key)
                if val is not None:
                    val = val.numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                    views[key][i] = val[: views[key].shape[1]]
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
