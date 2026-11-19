"""Latency of the fused NVLink all-reduce + residual + RMS norm against NCCL + rms_norm, all ranks in step.

    python bench/bench_allreduce.py --gpus 4

The custom kernel runs as a CUDA graph of back-to-back calls, replayed twice; the second replay is timed, so the
ranks are already in step (the first replay absorbs their start skew) and the time is the kernel latency without
the skew of a real model step. NCCL runs eagerly (launch overhead included).
"""

from __future__ import annotations

import argparse
import os


def rank_main(rank: int, world: int, port: int, rows_list: list[int], iters: int, queue) -> None:
    import torch

    from seventy.engine import distributed
    from seventy.engine.allreduce import CustomAllReduce
    from seventy.ops import kernels

    torch.cuda.set_device(rank)
    distributed.init(rank, world, port, "nccl")
    dev = torch.device("cuda", rank)
    ar = CustomAllReduce(rank, world, dev)

    w = torch.ones(5120, device=dev)
    out = {}
    for rows in rows_list:
        x = torch.randn(rows, 5120, device=dev).half()
        res = torch.randn(rows, 5120, device=dev).half()
        for name in ("custom", "nccl"):
            def body():
                for _ in range(iters):
                    if name == "custom":
                        ar.norm(x, res, w, 1e-6)
                    else:
                        y = x.clone()
                        torch.distributed.all_reduce(y)
                        kernels.rms_norm(y, w, 1e-6, res)
            for _ in range(2):
                body()
            torch.cuda.synchronize()
            g = None
            if name == "custom":
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    body()
                torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if g is not None:
                g.replay()
            s.record()
            if g is not None:
                g.replay()
            else:
                body()
            e.record()
            torch.cuda.synchronize()
            out[(rows, name)] = s.elapsed_time(e) * 1000 / iters
    queue.put((rank, out))
    distributed.shutdown()


def main() -> None:
    import torch.multiprocessing as mp

    from seventy.engine import distributed

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=rank_main, args=(r, a.gpus, port, a.rows, a.iters, q)) for r in range(a.gpus)]
    for p in procs:
        p.start()
    results = dict(q.get(timeout=600) for _ in procs)
    for p in procs:
        p.join(60)
    print(f"{'rows':>5} {'custom us':>10} {'nccl+norm us':>13}")
    for rows in a.rows:
        c = max(results[r][(rows, "custom")] for r in results)
        n = max(results[r][(rows, "nccl")] for r in results)
        print(f"{rows:>5} {c:>10.1f} {n:>13.1f}")


if __name__ == "__main__":
    main()
