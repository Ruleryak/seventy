"""The QSA kernels (csrc/qsa.cu) on a paged cache against the PyTorch definitions: pooled block keys, the positions
each query attends to (all visible ones up to the budget; past it, the best blocks and the unfinished last block),
and the attention over them with the output gate."""

import math

import pytest
import torch

from seventy import ops
from seventy.ops import kernels_available

pytestmark = pytest.mark.gpu
if not (torch.cuda.is_available() and kernels_available()):
    pytestmark = [pytestmark, pytest.mark.skip(reason="needs a CUDA GPU and the kernel library")]

BS, C, TOP, HEADS, DIM = 16, 4, 8, 4, 128
HQ, HKV, D = 8, 2, 256


@pytest.mark.parametrize("ctx,new", [(20, 4), (37, 1), (90, 3), (200, 5)])
def test_qsa_kernels_match_reference(ctx, new):
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(ctx + new)
    dev = "cuda"
    total = ctx + new
    blocks = -(-total // BS) + 1
    table = torch.randperm(blocks + 3, generator=g)[:blocks].int().to(dev)
    perm = torch.arange(total)
    slot = (table.cpu()[perm // BS].long() * BS + perm % BS).to(dev)
    nblk = blocks + 4
    idx_k = torch.zeros(nblk, BS, DIM, dtype=torch.float16, device=dev)
    raw = torch.randn(total, DIM, generator=g).half().to(dev)
    idx_k.view(-1, DIM)[slot] = raw
    positions = torch.zeros(nblk, BS, 3, dtype=torch.int32, device=dev)
    positions.view(-1, 3)[slot] = perm.int().to(dev)[:, None].expand(-1, 3).contiguous()
    pooled = torch.zeros(nblk, BS // C, DIM, dtype=torch.float16, device=dev)
    k_norm = (1 + torch.randn(DIM, generator=g) * 0.1).to(dev)
    half_rot = 16
    inv_freq = (1.0 / (1e7 ** (torch.arange(0, half_rot, dtype=torch.float64) * 2 / 32))).float().to(dev)
    section = (6, 5, 5)
    qs = torch.tensor([0, total], dtype=torch.int32, device=dev)
    zero = torch.tensor([0], dtype=torch.int32, device=dev)
    kernels.qsa_pool(pooled, idx_k, positions, table[None], qs, zero, total, k_norm, inv_freq, section, 32, C, 1e-6)
    nb = total // C
    keys = raw[: nb * C].float().view(nb, C, DIM).mean(1).half().float()
    keys = (keys * torch.rsqrt(keys.pow(2).mean(-1, keepdim=True) + 1e-6) * k_norm).half().float()
    pos3 = (torch.arange(nb) * C).to(dev)[None].expand(3, -1)
    cos, sin = ops.rotary_cos_sin(pos3, inv_freq, section)
    want_pooled = ops.apply_rotary(keys[:, None], cos, sin)[:, 0]
    starts = torch.arange(nb, device=dev) * C
    got_pooled = pooled.view(-1, DIM)[(table[starts // BS].long() * (BS // C) + (starts % BS) // C)]
    torch.testing.assert_close(got_pooled.float(), want_pooled, rtol=2e-3, atol=2e-3)

    q = torch.randn(new, HEADS, DIM, generator=g).half().to(dev)
    qs_new = torch.tensor([0, new], dtype=torch.int32, device=dev)
    ctx_t = torch.tensor([ctx], dtype=torch.int32, device=dev)
    sel = kernels.qsa_select(q, pooled, table[None], qs_new, ctx_t, C, TOP, total // C + 1, BS)
    scores = torch.relu(torch.einsum("qhd,bd->qbh", q.float(), got_pooled.float())).sum(-1) / math.sqrt(DIM)
    for i in range(new):
        visible = ctx + i + 1
        got = set(v for v in sel[i].tolist() if v >= 0)
        nbi = visible // C
        if nbi <= TOP:
            assert got == set(range(visible))
            continue
        best = torch.topk(scores[i, :nbi], TOP).indices.tolist()
        want = {b * C + j for b in best for j in range(C)} | set(range(nbi * C, visible))
        assert got == want, i

    k_cache = torch.randn(nblk, BS, HKV, D, generator=g).half().to(dev)
    v_cache = torch.randn(nblk, BS, HKV, D, generator=g).half().to(dev)
    qa = torch.randn(new, HQ, D, generator=g).half().to(dev)
    qkv = torch.randn(new, HQ * 2 * D, generator=g).half().to(dev)
    out = kernels.qsa_attention(qa, qkv, k_cache, v_cache, None, None, sel, table[None], qs_new, 1 / 16)
    for i in range(new):
        p = torch.tensor([v for v in sel[i].tolist() if v >= 0], device=dev)
        s = (table[p // BS].long() * BS + p % BS)
        kk = k_cache.view(-1, HKV, D)[s].float().repeat_interleave(HQ // HKV, dim=1)
        vv = v_cache.view(-1, HKV, D)[s].float().repeat_interleave(HQ // HKV, dim=1)
        att = torch.softmax(torch.einsum("hd,khd->hk", qa[i].float(), kk) / 16, dim=-1)
        o = torch.einsum("hk,khd->hd", att, vv).half().float()
        gate = qkv[i].view(HQ, 2, D)[:, 1].float()
        want = (o * torch.sigmoid(gate)).reshape(-1)
        torch.testing.assert_close(out[i].float(), want, rtol=5e-3, atol=5e-3)
