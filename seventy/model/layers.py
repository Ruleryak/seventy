"""Linear layers, built from one or more checkpoint modules and sliced for tensor parallel at load."""

from __future__ import annotations

import os

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import nvfp4, ops
from ..checkpoint import Checkpoint
from ..nvfp4 import GemmWeight, Nvfp4Weight
from ..ops import kernels
from ..upload import to_device

SMALL_BATCH = 64
FP8_LINEAR = os.environ.get("SEVENTY_FP8_LINEAR", "1") != "0"
MXFP8_ROWS = os.environ.get("SEVENTY_MXFP8_ROWS", "1") != "0"


@dataclass(frozen=True)
class Source:
    """Rows of one checkpoint module that go into a linear layer, in order. None means all rows."""

    module: str
    rows: tuple[tuple[int, int], ...] | None = None


class Linear:
    """y = x W^T, where W stacks the row ranges of all sources (so fused projections are one product).

    cols restricts W to an input column range (the shard of a row-parallel layer). NVFP4 weights stay packed;
    on the GPU, batches of up to SMALL_BATCH rows use the fused NVFP4 GEMM kernel, and larger batches (prefill)
    dequantize the weight to FP16 and use cuBLAS. On the CPU (reference runs) the weight is dequantized at each call.
    quantize rounds dense modules to NVFP4 on the GPU (lossy: for drafters), one global scale per source row range.
    Dense FP16 modules of checkpoints that ask for it (pack_dense: GGUF, whose dense tensors arrive dequantized) are
    kept packed for the dense GEMM kernel, which small batches use instead of cuBLAS; prefill unpacks them.
    """

    def __init__(
        self,
        ckpt: Checkpoint,
        sources: list[Source],
        dtype: torch.dtype,
        device: torch.device | str,
        cols: tuple[int, int] | None = None,
        keep_dense: bool = False,
        quantize: bool = False,
    ):
        self.dtype = dtype
        self.parts: list[Nvfp4Weight] = []
        self.dense: torch.Tensor | None = None
        self.packed = None
        self.gemm: GemmWeight | None = None
        quantized = [ckpt.is_nvfp4(s.module) for s in sources]
        if any(quantized) and not all(quantized):
            raise ValueError(f"cannot fuse quantized and dense modules: {[s.module for s in sources]}")
        self.fp8 = self.fp8_tail = None
        if (not quantized[0] and torch.device(device).type == "cuda" and dtype == torch.float16 and not keep_dense
                and not quantize and FP8_LINEAR and self._fp8(ckpt, sources, device, cols)):
            self.out_features = self.fp8.n + (self.fp8_tail.n if self.fp8_tail is not None else 0)
            return
        pieces = []
        for s in sources:
            n = ckpt.out_rows(s.module)
            for r in s.rows or ((0, n),):
                if quantized[0]:
                    self.parts.append(ckpt.nvfp4_part(s.module, rows=r, cols=cols).to(device))
                else:
                    pieces.append(ckpt.linear_part(s.module, rows=r, cols=cols))
        if pieces and quantize and torch.device(device).type == "cuda" and dtype == torch.float16 \
                and pieces[0].shape[1] % 64 == 0:
            self.gemm = nvfp4.pack_for_gemm([nvfp4.quantize(to_device(_to_dtype(p, dtype), device)) for p in pieces])
        elif pieces:
            self.dense = to_device(_to_dtype(torch.cat(pieces), dtype), device)
            if (getattr(ckpt, "pack_dense", False) and self.dense.is_cuda and dtype == torch.float16
                    and self.dense.shape[1] % 64 == 0):
                self.packed = kernels.DenseWeight(self.dense)
                self.dense = None
        elif keep_dense:
            self.dense = self._dequantize()
            self.parts = []
        elif self.parts[0].packed.is_cuda and dtype == torch.float16 and self.parts[0].shape[1] % 64 == 0:
            self.gemm = nvfp4.pack_for_gemm(self.parts)
            self.parts = []
        self.out_features = (self.dense.shape[0] if self.dense is not None else self.packed.n if self.packed is not None
                             else self.gemm.n if self.gemm is not None else sum(p.shape[0] for p in self.parts))

    def _fp8(self, ckpt: Checkpoint, sources: list[Source], device, cols) -> bool:
        """Keep FP8 modules (one scale per tensor or per row; MXFP8 modules with one scale per row, see
        Checkpoint.mxfp8_rows) as FP8 for the FP8 GEMM: the leading sources that are FP8 form one weight, and dense sources after them a small FP16 tail (in the Gated DeltaNet projection,
        the gate rows b and a). False, and nothing kept, when no leading source is FP8 or k is not a multiple
        of 64."""
        fp8 = ("fp8", "mxfp8") if MXFP8_ROWS else ("fp8",)
        kinds = [ckpt.kind(s.module) for s in sources]
        lead = 0
        while lead < len(kinds) and kinds[lead] in fp8:
            lead += 1
        if not lead or any(k in fp8 for k in kinds[lead:]):
            return False
        values, scales = [], []
        for s, kind in zip(sources[:lead], kinds):
            n = ckpt.out_rows(s.module)
            for r in s.rows or ((0, n),):
                part = ckpt.fp8_part if kind == "fp8" else ckpt.mxfp8_rows
                v, sc = part(s.module, rows=r, cols=cols)
                values.append(v)
                scales.append(sc)
        if values[0].shape[1] % 64:
            return False
        self.fp8 = kernels.DenseWeightFP8.from_fp8(to_device(torch.cat(values), device), torch.cat(scales))
        tail = []
        for s in sources[lead:]:
            n = ckpt.out_rows(s.module)
            for r in s.rows or ((0, n),):
                tail.append(ckpt.linear_part(s.module, rows=r, cols=cols))
        if tail:
            self.fp8_tail = kernels.DenseWeight(to_device(_to_dtype(torch.cat(tail), torch.float16), device))
        return True

    def _dequantize(self) -> torch.Tensor:
        if self.parts[0].packed.is_cuda and self.dtype == torch.float16:
            return torch.cat([ops.nvfp4_dequantize(p) for p in self.parts])
        return torch.cat([nvfp4.dequantize(p, self.dtype) for p in self.parts])

    def weight(self) -> torch.Tensor:
        if self.dense is not None:
            return self.dense
        if self.packed is not None:
            return kernels.dense_unpack(self.packed)
        if self.fp8 is not None:
            w = self.fp8.dequantize().to(torch.float16)
            if self.fp8_tail is not None:
                w = torch.cat((w, kernels.dense_unpack(self.fp8_tail)))
            return w
        if self.gemm is not None:
            return kernels.nvfp4_gemm_dequantize(self.gemm)
        return self._dequantize()

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """x W^T, written to out when given (a contiguous [rows, out_features] tensor)."""
        if self.gemm is not None and 0 < x.shape[0] <= SMALL_BATCH and x.is_contiguous():
            return kernels.nvfp4_gemm(x, self.gemm, out=out)
        if self.packed is not None and 0 < x.shape[0] <= SMALL_BATCH:
            y = kernels.dense_gemm(x.contiguous(), self.packed, out_fp32=False)
            return y if out is None else out.copy_(y)
        if self.fp8 is not None and 0 < x.shape[0] <= SMALL_BATCH:
            x = x.contiguous()
            y = kernels.dense_gemm(x, self.fp8, out_fp32=False)
            if self.fp8_tail is not None:
                y = torch.cat((y, kernels.dense_gemm(x, self.fp8_tail, out_fp32=False)), dim=1)
            return y if out is None else out.copy_(y)
        y = F.linear(x, self.weight())
        return y if out is None else out.copy_(y)


def _to_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float16 and t.dtype == torch.bfloat16:
        peak = t.abs().max().item() if t.numel() else 0.0
        if peak > 65504.0:
            raise ValueError(f"a BF16 weight has values up to {peak:g}, beyond the FP16 range")
    return t.to(dtype)
