"""Thin wrappers that run our CUDA kernels on PyTorch tensors. Inputs must be CUDA tensors on the current device."""

from __future__ import annotations

import torch

from ..nvfp4 import Nvfp4Weight
from ._lib import check, lib


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


def _expect(t: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, not {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _expect(x, "x", torch.float16)
    _expect(weight, "weight", torch.float32)
    dim = x.shape[-1]
    if weight.numel() != dim:
        raise ValueError(f"weight has {weight.numel()} values for rows of {dim}")
    out = torch.empty_like(x) if out is None else out
    _expect(out, "out", torch.float16)
    if residual is not None:
        _expect(residual, "residual", torch.float16)
        residual_out = torch.empty_like(residual) if residual_out is None else residual_out
        _expect(residual_out, "residual_out", torch.float16)
    rows = x.numel() // dim if dim else 0
    code = lib().seventy_rms_norm(
        out.data_ptr(),
        residual_out.data_ptr() if residual is not None else None,
        x.data_ptr(),
        residual.data_ptr() if residual is not None else None,
        weight.data_ptr(),
        rows,
        dim,
        eps,
        _stream(x),
    )
    check("rms_norm", code)
    return out, (residual_out if residual is not None else None)


def nvfp4_dequantize(w: Nvfp4Weight, out: torch.Tensor | None = None) -> torch.Tensor:
    _expect(w.packed, "packed", torch.uint8)
    _expect(w.scales, "scales", torch.uint8)
    n, k = w.shape
    out = torch.empty((n, k), dtype=torch.float16, device=w.packed.device) if out is None else out
    _expect(out, "out", torch.float16)
    code = lib().seventy_nvfp4_dequantize(
        out.data_ptr(), w.packed.data_ptr(), w.scales.data_ptr(), w.global_scale, n, k, _stream(out)
    )
    check("nvfp4_dequantize", code)
    return out
