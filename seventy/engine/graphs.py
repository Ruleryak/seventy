"""CUDA graphs of decode steps: one new token for each of up to B sequences, for each batch size bucket B.

A captured step reads its inputs from static device tensors; every step copies the real values in and replays
the graph. Rows beyond the real batch point at the cache's scratch block and scratch state slot. Every
tensor-parallel rank captures the same buckets in the same order, because the graphs contain collectives.
"""

from __future__ import annotations

import os

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

    def run(self, payload: list[tuple]) -> torch.Tensor:
        """payload: (token_ids [1], context_len, block_table, state_slot, logits, rope delta, ()) per sequence.
        Returns the logits of the real rows."""
        n = len(payload)
        b = self.bucket(n)
        graph, st, logits = self.graphs[b]
        bs = self.cache.block_size
        scratch = self.cache.scratch_block
        tokens = torch.zeros(b, dtype=torch.long)
        pos = torch.zeros(b, dtype=torch.long)
        ctx = torch.zeros(b, dtype=torch.int32)
        slots = torch.full((b,), self.cache.scratch_slot, dtype=torch.int32)
        kv = torch.full((b,), scratch * bs, dtype=torch.int32)
        width = max(len(p[2]) for p in payload)
        table = torch.full((b, width), scratch, dtype=torch.int32)
        for i, (ids, c, blocks, slot, _, delta, _) in enumerate(payload):
            tokens[i] = ids[0]
            pos[i] = c + delta
            ctx[i] = c
            slots[i] = slot
            kv[i] = blocks[c // bs] * bs + c % bs
            table[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        st.token_ids.copy_(tokens, non_blocking=True)
        st.positions.copy_(pos.expand(3, b), non_blocking=True)
        st.context_lens_t.copy_(ctx, non_blocking=True)
        st.state_slots_t.copy_(slots, non_blocking=True)
        st.kv_slots_t.copy_(kv, non_blocking=True)
        st.block_tables[:, :width].copy_(table, non_blocking=True)
        graph.replay()
        return logits[:n]
