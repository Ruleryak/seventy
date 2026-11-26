"""Small-batch tensor-parallel all-reduce over NVLink peer memory, fused with residual add and RMS norm
(csrc/allreduce.cu). Larger messages use NCCL.

Some hosts report peer access between their GPUs but drop or delay the peer writes (seen on PCIe machines). The
kernels' waits are bounded, and before first use every rank runs each kernel on known data (check). If any rank
fails, no rank uses these kernels, and NCCL runs without peer-to-peer transfers (NCCL_P2P_DISABLE=1). The setup
and the check exchange data through the process group's TCP store, so NCCL makes its communicator only after the
decision. After the check, a wait that gives up (30 s) marks a word in host memory that all ranks map; the engine
reads it after each step (failure)."""

from __future__ import annotations

import ctypes
import itertools
import mmap
import os
import tempfile

import torch

from ..ops._lib import check, lib
from . import distributed

CHECK_TIMEOUT_NS = 5 * 10**9
RUN_TIMEOUT_NS = 30 * 10**9
SITES = {1: "FP16 push data", 2: "FP32 push data", 3: "one-shot flags", 4: "two-shot flags", 5: "two-shot gather",
         6: "FP32 flags", 7: "split attention parts"}
_instances = itertools.count()


class SharedPage:
    """4096 bytes of host memory that all ranks map: a file in /dev/shm (else the temporary directory), removed
    once every rank has mapped it."""

    def __init__(self, store, key: str, rank: int, world: int):
        if rank == 0:
            folder = "/dev/shm" if os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
            fd, path = tempfile.mkstemp(prefix="seventy-ar-", dir=folder)
            os.ftruncate(fd, mmap.PAGESIZE)
            store.set(f"{key}/page", path)
        else:
            path = store.get(f"{key}/page").decode()
            fd = os.open(path, os.O_RDWR)
        try:
            self.map = mmap.mmap(fd, mmap.PAGESIZE)
        finally:
            os.close(fd)
        self._view = ctypes.c_char.from_buffer(self.map)
        self.address = ctypes.addressof(self._view)
        store.set(f"{key}/mapped/{rank}", "1")
        if rank == 0:
            store.wait([f"{key}/mapped/{r}" for r in range(world)])
            os.unlink(path)

    def word(self) -> int:
        return int.from_bytes(self.map[0:4], "little")


def describe(word: int) -> str:
    site, rank = word >> 8, (word & 0xff) - 1
    return f"rank {rank} waited more than the time limit for {SITES.get(site, f'site {site}')}"


class CustomAllReduce:
    def __init__(self, rank: int, world: int, device: torch.device, max_rows: int = 64, dim: int = 5120):
        self.rank, self.world, self.max_rows, self.dim = rank, world, max_rows, dim
        self.device = torch.device(device)
        store = distributed.store()
        key = f"seventy/ar/{next(_instances)}"
        slot = -(-max_rows * dim * 4 // 32) * 32
        ptr = ctypes.c_void_p()
        handle = (ctypes.c_char * 64)()
        check("ar_alloc", lib().seventy_ar_alloc(world, slot, ctypes.byref(ptr), handle))
        store.set(f"{key}/handle/{rank}", bytes(handle))
        ptrs = []
        for r in range(world):
            if r == rank:
                ptrs.append(ptr.value)
                continue
            peer = ctypes.c_void_p()
            buf = (ctypes.c_char * 64).from_buffer_copy(store.get(f"{key}/handle/{r}"))
            check("ar_open", lib().seventy_ar_open(buf, ctypes.byref(peer)))
            ptrs.append(peer.value)
        arr = (ctypes.c_void_p * world)(*ptrs)
        ctx = ctypes.c_void_p()
        check("ar_context", lib().seventy_ar_context(rank, world, arr, slot, ctypes.byref(ctx)))
        self.ctx = ctx
        self.page = SharedPage(store, key, rank, world)
        check("ar_guard", lib().seventy_ar_guard(ctx, self.page.address, CHECK_TIMEOUT_NS))
        store.set(f"{key}/ready/{rank}", "1")
        store.wait([f"{key}/ready/{r}" for r in range(world)])
        problem = self.check()
        store.set(f"{key}/check/{rank}", problem)
        problems = {r: store.get(f"{key}/check/{r}").decode() for r in range(world)}
        bad = sorted((r, p) for r, p in problems.items() if p)
        self.problem = "; ".join(f"GPU {r}: {p}" for r, p in bad)
        self.usable = not bad
        if self.usable:
            check("ar_guard", lib().seventy_ar_guard(ctx, None, RUN_TIMEOUT_NS))
        else:
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    def check(self) -> str:
        """Run every kernel of this all-reduce on small integers (exact sums in any order) and compare: '' when all
        match, else what failed. Every rank must call it at the same time."""
        dev, world, dim = self.device, self.world, self.dim

        def inputs(rows: int, r: int, scale: int = 4):
            g = torch.Generator().manual_seed(rows * 131 + r)
            return torch.randint(-scale, scale + 1, (rows, dim), generator=g).float()

        try:
            for rows in sorted({1, min(8, self.max_rows), min(9, self.max_rows), self.max_rows}):
                xs = [inputs(rows, r) for r in range(world)]
                total = sum(xs)
                res0 = inputs(rows, 1000, 50)
                w = torch.full((dim,), 0.5)
                want_res = total + res0
                want_out = want_res * torch.rsqrt(want_res.pow(2).mean(-1, keepdim=True) + 1e-6) * w
                x16, w32 = xs[self.rank].half().to(dev), w.to(dev)
                got = self.reduce(x16).float().cpu()
                got32 = self.reduce(xs[self.rank].to(dev)).cpu()
                res = res0.half().to(dev)
                out, res = self.norm(x16, res, w32, 1e-6, True)
                out_flags, _ = self.norm(x16, None, w32, 1e-6)
                torch.cuda.synchronize(dev)
                word = self.page.word()
                if word:
                    return describe(word)
                if not torch.equal(got, total) or not torch.equal(got32, total):
                    return f"wrong all-reduce sums ({rows} rows)"
                if not torch.equal(res.float().cpu(), want_res):
                    return f"wrong residual after the all-reduce ({rows} rows)"
                if (out.float().cpu() - want_out).abs().max().item() > 1e-2:
                    return f"wrong RMS norm after the all-reduce ({rows} rows)"
                ref_flags = total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + 1e-6) * w
                if (out_flags.float().cpu() - ref_flags).abs().max().item() > 1e-2:
                    return f"wrong one-shot all-reduce ({rows} rows)"
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        word = self.page.word()
        return describe(word) if word else ""

    def failure(self) -> str | None:
        """Why a wait gave up since the check (on any rank), or None."""
        word = self.page.word()
        return describe(word) if word else None

    def fits(self, x: torch.Tensor) -> bool:
        return (x.is_cuda and x.dtype in (torch.float16, torch.float32) and x.dim() == 2 and x.shape[-1] == self.dim
                and x.shape[0] <= self.max_rows)

    def norm(self, x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float,
             full_residual: bool = True):
        """(rms_norm(residual + all_reduce(x)) * weight, the new residual). FP16 rows with a residual take the
        two-shot kernel (faster than the one-shot kernel at every row count on V100), which updates the full
        residual on every rank; with SEVENTY_AR=home it updates only this rank's shard of rows unless full_residual
        is set, so callers keep treating the residual stream as sharded between calls."""
        x = x.contiguous()
        out = torch.empty_like(x)
        if residual is not None and x.dtype == torch.float16:
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

    def attn_merge(self, out: torch.Tensor, ws: torch.Tensor, qkv: torch.Tensor, rows: int, hq: int, splits: int,
                   own0: int, own: int, peer: int, peer_own0: int) -> torch.Tensor:
        """The split attention merge with the peer rank that shares this rank's KV head (seventy_ar_attn_merge):
        out FP16 [rows, own * 256] of the heads own0 .. own0 + own - 1 from this rank's seventy_qsa_attention
        workspace ws and the peer's; the peer's own heads start at peer_own0."""
        check("ar_attn_merge", lib().seventy_ar_attn_merge(
            self.ctx, out.data_ptr(), ws.data_ptr(), qkv.data_ptr(), qkv.stride(0), rows, hq, splits, own0, own, peer,
            peer_own0, torch.cuda.current_stream(out.device).cuda_stream))
        return out

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """all_reduce(x): FP16 (FP32 sum rounded once) or FP32."""
        x = x.contiguous()
        if x.dtype == torch.float32:
            out = torch.empty_like(x)
            check("ar_sum_f32", lib().seventy_ar_sum_f32(self.ctx, out.data_ptr(), x.data_ptr(), x.shape[0], x.shape[1],
                                                         torch.cuda.current_stream(x.device).cuda_stream))
            return out
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if os.environ.get("SEVENTY_AR") not in ("home", "flags"):
            code = lib().seventy_ar_norm_2shot(self.ctx, out.data_ptr(), None, x.data_ptr(), None, x.shape[0],
                                               x.shape[1], 0.0, 0, stream)
        else:
            code = lib().seventy_ar_norm(self.ctx, out.data_ptr(), None, x.data_ptr(), None, x.shape[0], x.shape[1],
                                         0.0, stream)
        check("ar_reduce", code)
        return out
