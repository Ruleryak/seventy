"""PyTorch reference implementations. They define what each kernel must compute, and they run every operation
on CPU tensors. The math is FP32 with the same rounding points as the kernels."""

from __future__ import annotations

import torch

from .. import nvfp4
from ..nvfp4 import Nvfp4Weight


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    s = x if residual is None else (x.float() + residual.float()).to(x.dtype)
    f = s.float()
    out = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return out.to(x.dtype), (None if residual is None else s)


def nvfp4_dequantize(w: Nvfp4Weight) -> torch.Tensor:
    return nvfp4.dequantize(w, torch.float16)
