"""Linear layers, built from one or more checkpoint modules and sliced for tensor parallel at load."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import nvfp4, ops
from ..checkpoint import Checkpoint
from ..nvfp4 import Nvfp4Weight


@dataclass(frozen=True)
class Source:
    """Rows of one checkpoint module that go into a linear layer, in order. None means all rows."""

    module: str
    rows: tuple[tuple[int, int], ...] | None = None


class Linear:
    """y = x W^T, where W stacks the row ranges of all sources (so fused projections are one product).

    cols restricts W to an input column range (the shard of a row-parallel layer). NVFP4 weights stay packed;
    this version dequantizes them at each call (the GEMM kernels replace that).
    """

    def __init__(
        self,
        ckpt: Checkpoint,
        sources: list[Source],
        dtype: torch.dtype,
        device: torch.device | str,
        cols: tuple[int, int] | None = None,
        keep_dense: bool = False,
    ):
        self.dtype = dtype
        self.parts: list[Nvfp4Weight] = []
        self.dense: torch.Tensor | None = None
        quantized = [ckpt.is_nvfp4(s.module) for s in sources]
        if any(quantized) and not all(quantized):
            raise ValueError(f"cannot fuse quantized and dense modules: {[s.module for s in sources]}")
        pieces = []
        for s in sources:
            n = ckpt.shape(f"{s.module}.weight_packed" if quantized[0] else f"{s.module}.weight")[0]
            for r in s.rows or ((0, n),):
                if quantized[0]:
                    self.parts.append(ckpt.nvfp4_part(s.module, rows=r, cols=cols).to(device))
                else:
                    pieces.append(ckpt.dense_part(f"{s.module}.weight", rows=r, cols=cols))
        if pieces:
            self.dense = _to_dtype(torch.cat(pieces), dtype).to(device)
        elif keep_dense:
            self.dense = self._dequantize()
            self.parts = []
        self.out_features = self.dense.shape[0] if self.dense is not None else sum(p.shape[0] for p in self.parts)

    def _dequantize(self) -> torch.Tensor:
        if self.parts[0].packed.is_cuda and self.dtype == torch.float16:
            return torch.cat([ops.nvfp4_dequantize(p) for p in self.parts])
        return torch.cat([nvfp4.dequantize(p, self.dtype) for p in self.parts])

    def weight(self) -> torch.Tensor:
        return self.dense if self.dense is not None else self._dequantize()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight())


def _to_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float16 and t.dtype == torch.bfloat16:
        peak = t.abs().max().item() if t.numel() else 0.0
        if peak > 65504.0:
            raise ValueError(f"a BF16 weight has values up to {peak:g}, beyond the FP16 range")
    return t.to(dtype)
