"""Time the NVFP4 GEMM kernel on the model's shapes, against the memory floor and cuBLAS on FP16 weights.

    python bench/bench_gemm.py [--tp 4] [--m 1 8 16 32 64]
"""

from __future__ import annotations

import argparse

import torch

from seventy import nvfp4
from seventy.ops import kernels

HIDDEN, INTER = 5120, 17408


def shapes(tp: int) -> dict[str, tuple[int, int]]:
    return {
        "gdn in_proj": ((2 * 2048 + 6144 + 6144 + 48 + 48) // tp, HIDDEN),
        "gdn out_proj": (HIDDEN, 6144 // tp),
        "attn qkv": ((24 * 512 + 2 * 4 * 256) // tp, HIDDEN),
        "attn o_proj": (HIDDEN, 6144 // tp),
        "mlp gate_up": (2 * INTER // tp, HIDDEN),
        "mlp down": (HIDDEN, INTER // tp),
    }


def timed(fn, iters: int = 50) -> float:
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (5 * iters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ap.add_argument("--bandwidth", type=float, default=850.0, help="GB/s used for the memory floor")
    ap.add_argument("--sweep", action="store_true", help="also try other split and warp counts, print the best")
    a = ap.parse_args()
    torch.manual_seed(0)
    print(f"TP{a.tp}   shape (n x k)        m   seventy us   floor us   GB/s   cuBLAS-FP16 us")
    for name, (n, k) in shapes(a.tp).items():
        part = nvfp4.quantize(torch.randn(n, k) * 0.02).to("cuda")
        w = nvfp4.pack_for_gemm([part])
        dense = kernels.nvfp4_dequantize(part)
        nbytes = w.wpk.numel() + w.wsc.numel()
        for m in a.m:
            x = torch.randn(m, k, device="cuda").half()
            out = torch.empty(m, n, device="cuda", dtype=torch.float16)
            ws = torch.zeros(kernels.lib().seventy_nvfp4_gemm_workspace(m, w.n_padded, 64) // 4 + 1, device="cuda")
            splits, warps = kernels.nvfp4_gemm_config(w, m, x.device)
            us = timed(lambda: kernels.nvfp4_gemm(x, w, splits=splits, warps=warps, out=out, workspace=ws))
            best = ""
            if a.sweep:
                trials = {}
                for wp in (2, 4, 8):
                    for sp in sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, 32}):
                        if sp <= k // 64:
                            trials[(sp, wp)] = timed(lambda: kernels.nvfp4_gemm(x, w, splits=sp, warps=wp, out=out,
                                                                                workspace=ws), iters=20)
                (bs, bw), bus = min(trials.items(), key=lambda kv: kv[1])
                best = f"   best {bus:6.1f} us at splits {bs}, warps {bw} (default {splits}, {warps})"
            ref = timed(lambda: torch.matmul(x, dense.T, out=out))
            floor = nbytes / (a.bandwidth * 1e3)
            print(f"{name:13} {n:6} x {k:5}  {m:3}  {us:10.1f}  {floor:9.1f}  {nbytes / us / 1e3:5.0f}  {ref:10.1f}{best}")


if __name__ == "__main__":
    main()
