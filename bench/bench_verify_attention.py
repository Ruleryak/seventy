"""Time the speculative verify attention kernel: 1 or 4 sequences, 8 new tokens each, one KV head, 6 query heads,
for several minimum split sizes."""

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
    for seqs in (1, 4):
        for ctx in (2800, 32000, 100000):
            per = -(-(ctx + tokens) // bs)
            blocks = per * seqs
            k = torch.randint(-127, 127, (blocks, bs, hkv, d), dtype=torch.int8, device="cuda")
            v = torch.randint(-127, 127, (blocks, bs, hkv, d), dtype=torch.int8, device="cuda")
            ks = torch.rand(blocks, bs, hkv, device="cuda") * 0.01
            vs = torch.rand(blocks, bs, hkv, device="cuda") * 0.01
            q = torch.randn(tokens * seqs, hq, d, device="cuda").half()
            qkv = torch.randn(tokens * seqs, 2 * hq * d + 2 * hkv * d, device="cuda").half()
            bt = torch.arange(blocks, dtype=torch.int32, device="cuda").view(seqs, per)
            qs = torch.arange(seqs + 1, dtype=torch.int32, device="cuda") * tokens
            cl = torch.full((seqs,), ctx, dtype=torch.int32, device="cuda")
            line = [f"{seqs} seq ctx {ctx:6d}:"]
            for msk in (128, 512, 1024, 2048):
                us = timed(lambda: kernels.decode_attention(q, qkv, k, v, bt, qs, cl, tokens, 0, 1 / math.sqrt(d), ks,
                                                            vs, min_split_keys=msk))
                line.append(f"{msk}: {us:6.1f}")
            gb = seqs * ctx * d * 2 * 1.0 / 1e9
            print("  ".join(line) + f"   (KV {gb * 1000:.0f} MB)", flush=True)
            del k, v


if __name__ == "__main__":
    main()
