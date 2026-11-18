"""Tensor parallel layout: which shard of each weight this process holds, and the collective operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Parallel:
    rank: int = 0
    size: int = 1

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


SINGLE = Parallel()
