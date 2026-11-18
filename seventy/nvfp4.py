"""The NVFP4 weight format ("nvfp4-pack-quantized" in compressed-tensors checkpoints).

A weight W with n output rows and k input columns is stored as:

- packed: uint8 [n, k / 2]. Two E2M1 values per byte; the even column is in the low nibble.
- scales: FP8 E4M3 [n, k / 16], kept here as their raw bytes (uint8). One scale per group of 16 columns.
- global_scale: one FP32 value per tensor.

W[i, j] = e2m1(packed, i, j) * e4m3(scales[i, j // 16]) / global_scale
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP = 16
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@dataclass
class Nvfp4Weight:
    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: float

    def __post_init__(self):
        n, half_k = self.packed.shape
        if self.packed.dtype != torch.uint8 or self.scales.dtype != torch.uint8:
            raise ValueError("packed values and scales must be uint8 tensors")
        if tuple(self.scales.shape) != (n, 2 * half_k // GROUP) or (2 * half_k) % GROUP:
            raise ValueError(f"scales {tuple(self.scales.shape)} do not match packed values {tuple(self.packed.shape)}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.packed.shape[0], 2 * self.packed.shape[1]

    def to(self, device) -> "Nvfp4Weight":
        return Nvfp4Weight(self.packed.to(device), self.scales.to(device), self.global_scale)

    def rows(self, start: int, stop: int) -> "Nvfp4Weight":
        return Nvfp4Weight(self.packed[start:stop].contiguous(), self.scales[start:stop].contiguous(), self.global_scale)

    def cols(self, start: int, stop: int) -> "Nvfp4Weight":
        if start % GROUP or stop % GROUP:
            raise ValueError(f"column slice [{start}, {stop}) must be aligned to groups of {GROUP}")
        return Nvfp4Weight(
            self.packed[:, start // 2 : stop // 2].contiguous(),
            self.scales[:, start // GROUP : stop // GROUP].contiguous(),
            self.global_scale,
        )


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """E2M1 values of a packed uint8 [n, k / 2] tensor as FP32 [n, k]."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    return torch.stack((table[low], table[high]), dim=-1).reshape(packed.shape[0], -1)


def scale_values(scales: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 scale bytes as FP32."""
    return scales.view(torch.float8_e4m3fn).float()


def dequantize(w: Nvfp4Weight, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """The dense weight [n, k]. The math is FP32 with one final rounding to dtype, as in the kernels."""
    values = unpack(w.packed)
    scale = scale_values(w.scales) / torch.tensor(w.global_scale, dtype=torch.float32)
    n, k = values.shape
    return (values.view(n, k // GROUP, GROUP) * scale.unsqueeze(-1)).view(n, k).to(dtype)


def pack(values: torch.Tensor) -> torch.Tensor:
    """Pack FP32 values that are exactly E2M1 numbers into uint8 [n, k / 2]."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=values.device)
    match = values.unsqueeze(-1) == table
    match[..., 0] &= ~torch.signbit(values)
    match[..., 8] &= torch.signbit(values)
    if not bool(match.any(-1).all()):
        raise ValueError("values are not all E2M1 numbers")
    codes = match.int().argmax(-1).to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def quantize(weight: torch.Tensor) -> Nvfp4Weight:
    """Round a dense FP32 weight to NVFP4 (round to nearest). Used to make test data."""
    n, k = weight.shape
    if k % GROUP:
        raise ValueError(f"k = {k} is not a multiple of {GROUP}")
    w = weight.float().view(n, k // GROUP, GROUP)
    amax = w.abs().amax()
    global_scale = float(448.0 * 6.0 / amax) if amax > 0 else 1.0
    block = (w.abs().amax(-1) / 6.0 * global_scale).clamp(max=448.0)
    scales = block.to(torch.float8_e4m3fn)
    eff = scales.float() / global_scale
    scaled = torch.where(eff.unsqueeze(-1) > 0, w / eff.unsqueeze(-1).clamp(min=1e-30), torch.zeros_like(w))
    mags = torch.tensor(E2M1_VALUES[:8], dtype=torch.float32)
    nearest = (scaled.abs().unsqueeze(-1) - mags).abs().argmin(-1)
    q = mags[nearest] * torch.where(scaled < 0, -1.0, 1.0)
    return Nvfp4Weight(pack(q.view(n, k)), scales.view(torch.uint8).contiguous(), global_scale)
