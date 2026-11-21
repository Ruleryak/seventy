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
        if self.custom is not None and x.dim() == 2 and self.custom.fits(x):
            return self.custom.reduce(x)
        torch.distributed.all_reduce(x)
        return x


    def reduce_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None,
                    full_residual: bool = False):
        """(rms_norm(residual + all_reduce(x)) * weight, new residual): x holds this rank's partial sums. The new
        residual is complete on every rank with the default custom all-reduce; with SEVENTY_AR=home it may be correct
        only for this rank's shard of rows unless full_residual is set. The normed output is always complete. Callers
        pass the residual on to the next reduce_norm and set full_residual where they read it."""
        from . import ops

        if (self.size > 1 and self.custom is not None and residual is not None and x.dtype == torch.float16
                and self.custom.fits(x)):
            return self.custom.norm(x, residual, weight, eps, full_residual)
        return ops.rms_norm(self.all_reduce(x), weight, eps, residual)


SINGLE = Parallel()
