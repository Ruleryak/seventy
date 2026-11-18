"""Time the speculative verify attention kernel: one sequence, 8 new tokens, one KV head, 6 query heads."""

import math
import sys

import torch

from seventy.ops import kernels
from seventy.ops import reference as R


def timed(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main():
    hq, hkv, d, bs, tokens = 6, 1, 256, 16, 8
    for ctx in (2800, 32000, 100000):
        blocks = -(-(ctx + tokens) // bs)
        k = torch.randint(-127, 127, (blocks, bs, hkv, d), dtype=torch.int8, device="cuda")
        v = torch.randint(-127, 127, (blocks, bs, hkv, d), dtype=torch.int8, device="cuda")
        ks = torch.rand(blocks, bs, hkv, device="cuda") * 0.01
        vs = torch.rand(blocks, bs, hkv, device="cuda") * 0.01
        q = torch.randn(tokens, hq, d, device="cuda").half()
        qkv = torch.randn(tokens, 2 * hq * d + 2 * hkv * d, device="cuda").half()
        bt = torch.arange(blocks, dtype=torch.int32, device="cuda")[None]
        qs = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        cl = torch.tensor([ctx], dtype=torch.int32, device="cuda")
        line = [f"ctx {ctx:6d}:"]
        for msk in (64, 128, 256, 512):
            us = timed(lambda: kernels.decode_attention(q, qkv, k, v, bt, qs, cl, tokens, 0, 1 / math.sqrt(d), ks, vs,
                                                        min_split_keys=msk))
            line.append(f"min {msk}: {us:6.1f} us")
        print("  ".join(line))


if __name__ == "__main__":
    main()
