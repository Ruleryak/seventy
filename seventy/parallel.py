"""Tensor parallel layout: which shard of each weight this process holds, and the collective operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Parallel:
    rank: int = 0
    size: int = 1
    custom: object = None

    def span(self, total: int) -> tuple[int, int]:
        """This rank's [start, stop) of `total` items split evenly over the ranks."""
        if total % self.size:
            raise ValueError(f"{total} does not split evenly over {self.size} GPUs")
        part = total // self.size
        return self.rank * part, (self.rank + 1) * part

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return x
        torch.distributed.all_reduce(x)
        return x


    def reduce_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None):
        """(rms_norm(residual + all_reduce(x)) * weight, new residual): x holds this rank's partial sums."""
        from . import ops

        if self.size > 1 and self.custom is not None and residual is not None and self.custom.fits(x):
            return self.custom.norm(x, residual, weight, eps)
        return ops.rms_norm(self.all_reduce(x), weight, eps, residual)


SINGLE = Parallel()
