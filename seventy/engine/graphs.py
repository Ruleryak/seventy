"""CUDA graphs of decode steps: one new token for each of up to B sequences, for each batch size bucket B.

A captured step reads its inputs from static device tensors; every step copies the real values in and replays
the graph. Rows beyond the real batch point at the cache's scratch block and scratch state slot. Every
tensor-parallel rank captures the same buckets in the same order, because the graphs contain collectives.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .cache import ModelCache
from .step import StepInput

BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def spec_buckets(limit: int, mode: str | None = None) -> tuple[int, ...]:
    """Batch sizes that get a graph of a whole speculative step, up to limit sequences: every size (mode "all", the
    default) or the powers of two and limit (mode "pow2"; SEVENTY_SPEC_BUCKETS sets the mode). A batch replays the
    smallest graph that holds it. A speculative step computes a block of rows for each sequence (8 with DFlash2), and
    at those row counts the verify GEMMs cost about as much as the rows they compute: with only the powers of two, 5
    sequences ran in the graph of 8 and took almost its time (44.4 against 47.5 ms at 100K tokens of context)."""
    if (mode or os.environ.get("SEVENTY_SPEC_BUCKETS", "all")) == "pow2":
        return tuple(b for b in BUCKETS if b <= limit) + (() if limit in BUCKETS else (limit,))
    return tuple(range(1, limit + 1))


class DecodeGraphs:
    def __init__(self, model, cache: ModelCache, device: torch.device, max_blocks_per_seq: int,
                 buckets: tuple[int, ...] = BUCKETS):
        self.model = model
        self.cache = cache
        self.device = device
        self.width = max_blocks_per_seq
        self.buckets = tuple(sorted(b for b in buckets if b <= cache.num_slots + 64))
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, StepInput, torch.Tensor]] = {}
        self.sent: dict[int, tuple] = {}
        self.slot_tables: dict[int, np.ndarray] = {}
        self.slot_len: dict[int, int] = {}

    def _static_step(self, b: int) -> StepInput:
        dev = self.device
        scratch = self.cache.scratch_block
        return StepInput(
            token_ids=torch.zeros(b, dtype=torch.long, device=dev),
            positions=torch.zeros(3, b, dtype=torch.long, device=dev),
            block_tables=torch.full((b, self.width), scratch, dtype=torch.int32, device=dev),
            logit_rows=torch.arange(b, dtype=torch.long, device=dev),
            query_start=list(range(b + 1)),
            context_lens=[0] * b,
            state_slots=[self.cache.scratch_slot] * b,
            block_size=self.cache.block_size,
            query_start_t=torch.arange(b + 1, dtype=torch.int32, device=dev),
            context_lens_t=torch.zeros(b, dtype=torch.int32, device=dev),
            state_slots_t=torch.full((b,), self.cache.scratch_slot, dtype=torch.int32, device=dev),
            kv_slots_t=torch.full((b,), scratch * self.cache.block_size, dtype=torch.int32, device=dev),
            max_blocks=0,
        )

    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        for b in sorted(self.buckets, reverse=True):
            static = self._static_step(b)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.model.forward(static, self.cache)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                logits, _ = self.model.forward(static, self.cache)
            self.graphs[b] = (graph, static, logits)
        torch.cuda.synchronize(self.device)

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def table_updates(self, rows: list[tuple]) -> list[tuple]:
        """(slot, first, total, ids) for each (slot, blocks, table_key) whose block table differs from what the
        ranks have: the appended blocks when the slot holds the same sequence with the same block list (table_key:
        sequence id, blocks epoch), else the whole table (as SpecGraphs._table_updates)."""
        updates = []
        for slot, blocks, key in rows:
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

    def run(self, payload: list[tuple], updates: list) -> torch.Tensor:
        """payload: (token id, context_len, state_slot, rope delta) per sequence; updates: table_updates of the
        sequences' block tables (every rank keeps per-slot tables). Returns the logits of the real rows."""
        n = len(payload)
        b = self.bucket(n)
        graph, st, logits = self.graphs[b]
        bs = self.cache.block_size
        scratch = self.cache.scratch_block
        self._apply_tables(updates)
        ids = np.zeros(b, dtype=np.int64)
        pos = np.zeros(b, dtype=np.int64)
        ctx = np.zeros(b, dtype=np.int32)
        slots = np.full(b, self.cache.scratch_slot, dtype=np.int32)
        kv = np.full(b, scratch * bs, dtype=np.int32)
        width = max(self.slot_len[p[2]] for p in payload)
        table = np.full((b, width), scratch, dtype=np.int32)
        for i, (tok, c, slot, delta) in enumerate(payload):
            t = self.slot_tables[slot]
            ids[i], pos[i], ctx[i], slots[i] = tok, c + delta, c, slot
            kv[i] = t[c // bs] * bs + c % bs
            table[i, : self.slot_len[slot]] = t[: self.slot_len[slot]]
        st.token_ids.copy_(torch.from_numpy(ids), non_blocking=True)
        st.positions.copy_(torch.from_numpy(pos).expand(3, b), non_blocking=True)
        st.context_lens_t.copy_(torch.from_numpy(ctx), non_blocking=True)
        st.state_slots_t.copy_(torch.from_numpy(slots), non_blocking=True)
        st.kv_slots_t.copy_(torch.from_numpy(kv), non_blocking=True)
        st.block_tables[:, :width].copy_(torch.from_numpy(table), non_blocking=True)
        graph.replay()
        return logits[:n]
