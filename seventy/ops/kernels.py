"""Thin wrappers that run our CUDA kernels on PyTorch tensors. Inputs must be CUDA tensors on the current device."""

from __future__ import annotations

import torch

from ..nvfp4 import GemmWeight, Nvfp4Weight
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


_SMS: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def nvfp4_gemm_config(w: GemmWeight, m: int, device: torch.device) -> tuple[int, int]:
    """(k splits, warps per block) for a weight and batch size, from sweeps on V100 (bench/bench_gemm.py --sweep):
    about 1100 warps in flight up to 16 rows, about 900 above, and at least 4 (up to 16 rows) or 8 k steps per split."""
    tiles = w.n_padded // 32
    ksteps = w.k // 64
    small = m <= 16
    warps = 8 if small else 4
    splits = round((1088 if small else 900) / tiles)
    return max(1, min(splits, max(1, ksteps // (4 if small else 8)))), warps


def nvfp4_gemm(x: torch.Tensor, w: GemmWeight, splits: int | None = None, warps: int | None = None,
               out: torch.Tensor | None = None, workspace: torch.Tensor | None = None) -> torch.Tensor:
    _expect(x, "x", torch.float16)
    m, k = x.shape
    if k != w.k:
        raise ValueError(f"x has k = {k}, the weight has k = {w.k}")
    auto_splits, auto_warps = nvfp4_gemm_config(w, m, x.device)
    splits = auto_splits if splits is None else splits
    warps = auto_warps if warps is None else warps
    need = lib().seventy_nvfp4_gemm_workspace(m, w.n_padded, splits)
    if need and (workspace is None or workspace.numel() * workspace.element_size() < need):
        workspace = torch.zeros(need // 4 + 1, dtype=torch.float32, device=x.device)
    out = torch.empty(m, w.n, dtype=torch.float16, device=x.device) if out is None else out
    code = lib().seventy_nvfp4_gemm(
        out.data_ptr(), x.data_ptr(), w.wpk.data_ptr(), w.wsc.data_ptr(), w.alpha.data_ptr(), m, w.n, w.n_padded,
        k, splits, warps, workspace.data_ptr() if need else None,
        workspace.numel() * workspace.element_size() if need else 0, _stream(x),
    )
    check("nvfp4_gemm", code)
    return out


def nvfp4_gemm_dequantize(w: GemmWeight, out: torch.Tensor | None = None) -> torch.Tensor:
    """The dense FP16 weight [n, k] of a repacked weight."""
    out = torch.empty(w.n, w.k, dtype=torch.float16, device=w.wpk.device) if out is None else out
    _expect(out, "out", torch.float16)
    code = lib().seventy_nvfp4_gemm_dequantize(out.data_ptr(), w.wpk.data_ptr(), w.wsc.data_ptr(), w.alpha.data_ptr(),
                                               w.n, w.n_padded, w.k, _stream(out))
    check("nvfp4_gemm_dequantize", code)
    return out
