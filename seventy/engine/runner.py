"""Runs scheduled work on the model: builds the step input, runs the forward pass, returns the logits."""

from __future__ import annotations

import torch

from ..engine.cache import ModelCache
from .scheduler import Work
from .step import SequenceChunk, StepInput


class ModelRunner:
    def __init__(self, model, cache: ModelCache, device: torch.device):
        self.model = model
        self.cache = cache
        self.device = device

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
        step = StepInput.build(chunks, self.cache.block_size, self.device)
        logits, _ = self.model.forward(step, self.cache)
        return logits
