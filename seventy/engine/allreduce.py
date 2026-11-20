"""Small-batch tensor-parallel all-reduce over NVLink peer memory, fused with residual add and RMS norm
(csrc/allreduce.cu). Larger messages use NCCL."""

from __future__ import annotations

import ctypes
import os

import torch
import torch.distributed as dist

from ..ops._lib import check, lib


TWO_SHOT_MIN = int(os.environ.get("SEVENTY_AR_TWO_SHOT_MIN", "16"))


class CustomAllReduce:
    def __init__(self, rank: int, world: int, device: torch.device, max_rows: int = 64, dim: int = 5120):
        self.rank, self.world, self.max_rows, self.dim = rank, world, max_rows, dim
        slot = -(-max_rows * dim * 4 // 32) * 32
        ptr = ctypes.c_void_p()
        handle = (ctypes.c_char * 64)()
        check("ar_alloc", lib().seventy_ar_alloc(world, slot, ctypes.byref(ptr), handle))
        handles = [None] * world
        dist.all_gather_object(handles, bytes(handle))
        ptrs = []
        for r in range(world):
            if r == rank:
                ptrs.append(ptr.value)
                continue
            peer = ctypes.c_void_p()
            buf = (ctypes.c_char * 64).from_buffer_copy(handles[r])
            check("ar_open", lib().seventy_ar_open(buf, ctypes.byref(peer)))
            ptrs.append(peer.value)
        arr = (ctypes.c_void_p * world)(*ptrs)
        ctx = ctypes.c_void_p()
        check("ar_context", lib().seventy_ar_context(rank, world, arr, slot, ctypes.byref(ctx)))
        self.ctx = ctx
        dist.barrier()

    def fits(self, x: torch.Tensor) -> bool:
        return (x.is_cuda and x.dtype in (torch.float16, torch.float32) and x.dim() == 2 and x.shape[-1] == self.dim
                and x.shape[0] <= self.max_rows)

    def norm(self, x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float,
             full_residual: bool = True):
        """(rms_norm(residual + all_reduce(x)) * weight, the new residual). With a residual, from TWO_SHOT_MIN rows
        and for a single row the two-shot kernel runs (faster there on V100; 2 to 15 rows the one-shot kernel is):
        the residual is then updated only for this rank's shard of rows, unless full_residual is set; the caller
        keeps the residual stream sharded between calls."""
        x = x.contiguous()
        out = torch.empty_like(x)
        if residual is not None and x.dtype == torch.float16 and (x.shape[0] >= TWO_SHOT_MIN or x.shape[0] == 1):
            code = lib().seventy_ar_norm_2shot(self.ctx, out.data_ptr(), residual.data_ptr(), x.data_ptr(),
                                               weight.data_ptr(), x.shape[0], x.shape[1], eps, 1 if full_residual else 0,
                                               torch.cuda.current_stream(x.device).cuda_stream)
            check("ar_norm_2shot", code)
            return out, residual
        code = lib().seventy_ar_norm(self.ctx, out.data_ptr(), residual.data_ptr() if residual is not None else None,
                                     x.data_ptr(), weight.data_ptr(), x.shape[0], x.shape[1], eps,
                                     torch.cuda.current_stream(x.device).cuda_stream)
        check("ar_norm", code)
        return out, residual

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """all_reduce(x): FP16 (FP32 sum rounded once) or FP32."""
        x = x.contiguous()
        if x.dtype == torch.float32:
            out = torch.empty_like(x)
            check("ar_sum_f32", lib().seventy_ar_sum_f32(self.ctx, out.data_ptr(), x.data_ptr(), x.shape[0], x.shape[1],
                                                         torch.cuda.current_stream(x.device).cuda_stream))
            return out
        out = torch.empty_like(x)
        code = lib().seventy_ar_norm(self.ctx, out.data_ptr(), None, x.data_ptr(), None, x.shape[0], x.shape[1], 0.0,
                                     torch.cuda.current_stream(x.device).cuda_stream)
        check("ar_reduce", code)
        return out
