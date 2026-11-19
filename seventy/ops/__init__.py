"""The operations the models use. CUDA tensors run our kernels where one exists; CPU tensors and operations
without a kernel yet run the PyTorch reference in reference.py."""

from __future__ import annotations

import torch

from ..nvfp4 import Nvfp4Weight
from . import kernels, reference
from ._lib import KernelError
from ._lib import available as kernels_available
from .reference import (
    apply_rotary,
    causal_conv1d,
    gated_delta_rule,
    gated_rms_norm,
    gdn_gates,
    kv_slots,
    paged_attention,
    rotary_cos_sin,
    sigmoid_gate,
    write_kv,
)

__all__ = [
    "KernelError",
    "apply_rotary",
    "causal_conv1d",
    "gated_delta_rule",
    "gated_rms_norm",
    "gdn_gates",
    "kernels_available",
    "kv_slots",
    "nvfp4_dequantize",
    "paged_attention",
    "rms_norm",
    "rotary_cos_sin",
    "sigmoid_gate",
    "silu_mul",
    "write_kv",
]


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RMS norm of the last dimension. With a residual, the norm is of x + residual, which is also returned."""
    if x.is_cuda:
        return kernels.rms_norm(x, weight, eps, residual)
    return reference.rms_norm(x, weight, eps, residual)


def nvfp4_dequantize(w: Nvfp4Weight) -> torch.Tensor:
    if w.packed.is_cuda:
        return kernels.nvfp4_dequantize(w)
    return reference.nvfp4_dequantize(w)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, where gate_up holds [gate | up] along the last dimension."""
    if gate_up.is_cuda and gate_up.dtype == torch.float16 and gate_up.dim() == 2 and gate_up.is_contiguous() \
            and gate_up.shape[1] % 4 == 0:
        return kernels.silu_mul(gate_up)
    return reference.silu_mul(gate_up)
