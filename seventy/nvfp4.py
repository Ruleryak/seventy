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

from .upload import to_device

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
        return Nvfp4Weight(to_device(self.packed, device), to_device(self.scales, device), self.global_scale)

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


def quantize(weight: torch.Tensor, global_scale: float | None = None, rows: int = 1024) -> Nvfp4Weight:
    """Round a dense weight [n, k] to NVFP4 (round to nearest) on its device, rows at a time. global_scale defaults
    to 448 * 6 / max |weight|. Used to make test data and the drafters' NVFP4 copies of dense weights (lossy). The
    divisors are tensors on the weight's device: CUDA divides by a host number through its reciprocal, which rounds
    differently, and block scales exactly between two E4M3 values then round the other way than on the CPU."""
    n, k = weight.shape
    if k % GROUP:
        raise ValueError(f"k = {k} is not a multiple of {GROUP}")
    if global_scale is None:
        amax = max((weight[s : s + rows].float().abs().amax().cpu() for s in range(0, n, rows)),
                   default=torch.tensor(0.0), key=float)
        global_scale = float(448.0 * 6.0 / amax) if amax > 0 else 1.0
    dev = weight.device
    six = torch.tensor(6.0, device=dev)
    gs = torch.tensor(global_scale, dtype=torch.float32, device=dev)
    mags = torch.tensor(E2M1_VALUES[:8], dtype=torch.float32, device=dev)
    packed, scales = [], []
    for s in range(0, n, rows):
        w = weight[s : s + rows].float().view(-1, k // GROUP, GROUP)
        block = (w.abs().amax(-1) / six * gs).clamp(max=448.0)
        sc = block.to(torch.float8_e4m3fn)
        eff = sc.float() / gs
        scaled = torch.where(eff.unsqueeze(-1) > 0, w / eff.unsqueeze(-1).clamp(min=1e-30), torch.zeros_like(w))
        nearest = (scaled.abs().unsqueeze(-1) - mags).abs().argmin(-1)
        q = mags[nearest] * torch.where(scaled < 0, -1.0, 1.0)
        packed.append(pack(q.view(-1, k)))
        scales.append(sc.view(torch.uint8))
    return Nvfp4Weight(torch.cat(packed), torch.cat(scales).contiguous(), global_scale)


GEMM_TILE = 32
GEMM_ALPHA_SHIFT = 2.0**7
_NIBBLE_POS = (3, 7, 2, 6, 1, 5, 0, 4)


def _lane_columns() -> list[int]:
    """Column (within a 32-column tile) of each lane in the GEMM kernel's mma.m8n8k4 B fragments."""
    return [8 * ((lane >> 2) & 3) + (lane & 3) + 4 * (lane >= 16) for lane in range(32)]


@dataclass
class GemmWeight:
    """An NVFP4 weight repacked for the GEMM kernel (see csrc/nvfp4_gemm.cu).

    wpk: uint8 [n_padded / 32, k / 32, 32 lanes, 16 bytes]. Each lane's 16 bytes are 4 little-endian words of
    8 codes for k = 8 j .. 8 j + 7, at nibbles 3, 7, 2, 6, 1, 5, 0, 4.
    wsc: uint8 [n_padded / 32, k / 64, 32 lanes, 4 bytes], the E4M3 scales of the 4 groups of a 64-wide k block.
    alpha: FP32 [n_padded], 2^7 / global_scale of the row's source tensor (0 for padding rows).
    """

    wpk: torch.Tensor
    wsc: torch.Tensor
    alpha: torch.Tensor
    n: int
    k: int

    @property
    def n_padded(self) -> int:
        return self.alpha.numel()


def pack_for_gemm(parts: list[Nvfp4Weight], block_rows: int = 2048) -> GemmWeight:
    """Stack the rows of `parts` (same k) and repack them for the GEMM kernel, on the parts' device, in blocks of
    block_rows rows to bound the temporary memory."""
    k = parts[0].shape[1]
    if any(p.shape[1] != k for p in parts):
        raise ValueError("all parts of a fused weight must have the same k")
    if k % 64:
        raise ValueError(f"k = {k} is not a multiple of 64")
    device = parts[0].packed.device
    n = sum(p.shape[0] for p in parts)
    n_padded = -(-n // GEMM_TILE) * GEMM_TILE
    tiles, chunks = n_padded // GEMM_TILE, k // 32
    wpk = torch.empty(tiles, chunks, 32, 16, dtype=torch.uint8, device=device)
    wsc = torch.empty(tiles, k // 64, 32, 4, dtype=torch.uint8, device=device)
    alpha = torch.zeros(n_padded, dtype=torch.float32, device=device)
    starts, row = [], 0
    for p in parts:
        if bool(((p.scales & 0x80) != 0).any()) or bool((p.scales == 0x7F).any()):
            raise ValueError("NVFP4 scales must be non-negative and finite")
        alpha[row : row + p.shape[0]] = GEMM_ALPHA_SHIFT / p.global_scale
        starts.append(row)
        row += p.shape[0]
    lanes = torch.tensor(_lane_columns(), device=device)
    shifts = torch.tensor([4 * q for q in _NIBBLE_POS], device=device, dtype=torch.int32)
    block = max(GEMM_TILE, block_rows // GEMM_TILE * GEMM_TILE)
    for r0 in range(0, n_padded, block):
        r1 = min(n_padded, r0 + block)
        packed = torch.zeros(r1 - r0, k // 2, dtype=torch.uint8, device=device)
        scales = torch.zeros(r1 - r0, k // GROUP, dtype=torch.uint8, device=device)
        for p, start in zip(parts, starts):
            a, b = max(r0, start), min(r1, start + p.shape[0])
            if a < b:
                packed[a - r0 : b - r0] = p.packed[a - start : b - start]
                scales[a - r0 : b - r0] = p.scales[a - start : b - start]
        t = (r1 - r0) // GEMM_TILE
        codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(r1 - r0, k).to(torch.int32)
        codes = codes.view(t, GEMM_TILE, chunks, 4, 8)[:, lanes].permute(0, 2, 1, 3, 4)
        words = (codes << shifts).sum(-1, dtype=torch.int32)
        wpk[r0 // GEMM_TILE : r1 // GEMM_TILE] = words.contiguous().view(torch.uint8).view(t, chunks, 32, 16)
        wsc[r0 // GEMM_TILE : r1 // GEMM_TILE] = scales.view(t, GEMM_TILE, k // 64, 4)[:, lanes].permute(0, 2, 1, 3)
    return GemmWeight(wpk=wpk, wsc=wsc, alpha=alpha, n=n, k=k)
