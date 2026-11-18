"""Runs scheduled work on the model: builds the step input, sends it to the tensor-parallel workers, runs the
forward pass, and returns the logits (rank 0)."""

from __future__ import annotations

import pickle

import torch

from ..engine.cache import ModelCache
from .graphs import DecodeGraphs
from .scheduler import Work
from .step import SequenceChunk, StepInput


class ModelRunner:
    def __init__(self, model, cache: ModelCache, device: torch.device, peers=()):
        self.model = model
        self.cache = cache
        self.device = device
        self.peers = list(peers)
        self.graphs: DecodeGraphs | None = None

    def capture(self, max_blocks_per_seq: int, buckets: tuple[int, ...]) -> None:
        msg = pickle.dumps(("capture", max_blocks_per_seq, list(buckets)))
        for p in self.peers:
            p.send_bytes(msg)
        self.graphs = DecodeGraphs(self.model, self.cache, self.device, max_blocks_per_seq, buckets)
        self.graphs.capture()
        for p in self.peers:
            if not p.poll(900):
                raise RuntimeError("a tensor-parallel worker did not finish capturing CUDA graphs")
            reply = p.recv()
            if reply[0] != "captured":
                raise RuntimeError(f"a worker failed to capture CUDA graphs: {reply}")

    def run_decode(self, work: list[Work]) -> torch.Tensor:
        """Decode work (one new token per sequence) through the CUDA graphs when a bucket fits."""
        if self.graphs is None or self.graphs.bucket(len(work)) is None:
            return self.run(work)
        payload = [(w.request.tokens[w.start : w.start + 1], w.start, w.request.blocks, w.request.slot, "last")
                   for w in work]
        if self.peers:
            msg = pickle.dumps(("decode", payload))
            for p in self.peers:
                p.send_bytes(msg)
        return self.graphs.run(payload)

    def run(self, work: list[Work]) -> torch.Tensor:
        """FP32 logits for the rows each work item asked for ("last": its last token, "all": every token), in order."""
        chunks = [
            SequenceChunk(
                token_ids=w.request.tokens[w.start : w.start + w.count],
                context_len=w.start,
                block_table=w.request.blocks,
                state_slot=w.request.slot,
                logits=w.logits,
            )
            for w in work
        ]
        if self.peers:
            payload = pickle.dumps(("step", [(c.token_ids, c.context_len, c.block_table, c.state_slot, c.logits)
                                             for c in chunks]))
            for p in self.peers:
                p.send_bytes(payload)
        step = StepInput.build(chunks, self.cache.block_size, self.device)
        logits, _ = self.model.forward(step, self.cache)
        return logits

    def close(self) -> None:
        for p in self.peers:
            try:
                p.send_bytes(pickle.dumps(("exit",)))
            except (BrokenPipeError, EOFError, OSError):
                pass
