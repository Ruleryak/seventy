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
    raw = torch.randn(total, DIM, generator=g).half().to(dev)
    positions = torch.zeros(nblk, BS, 3, dtype=torch.int32, device=dev)
    positions.view(-1, 3)[slot] = perm.int().to(dev)[:, None].expand(-1, 3).contiguous()
    pooled = torch.zeros(nblk, BS // C, DIM, dtype=torch.float16, device=dev)
    k_norm = (1 + torch.randn(DIM, generator=g) * 0.1).to(dev)
    half_rot = 16
    inv_freq = (1.0 / (1e7 ** (torch.arange(0, half_rot, dtype=torch.float64) * 2 / 32))).float().to(dev)
    section = (6, 5, 5)
    ring = torch.zeros(3, 32, DIM, dtype=torch.float16, device=dev)
    slots_t = torch.tensor([1], dtype=torch.int32, device=dev)
    for a, b in ((0, ctx), (ctx, total)):
        if b == a:
            continue
        qs_ab = torch.tensor([0, b - a], dtype=torch.int32, device=dev)
        ctx_ab = torch.tensor([a], dtype=torch.int32, device=dev)
        kernels.qsa_pool(pooled, raw[a:b], ring, slots_t, positions, table[None], qs_ab, ctx_ab, b - a, k_norm,
                         inv_freq, section, 32, C, BS, 1e-6)
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


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("hq,hkv", [(4, 1), (6, 1), (12, 1), (24, 2), (24, 1)])
@pytest.mark.parametrize("splits", [None, 1, 5])
def test_qsa_attention_over_any_selection(int8, hq, hkv, splits):
    """seventy_qsa_attention for any group size up to 32, split count and selection with holes (-1), on FP16 and
    INT8 caches (per token and KV head scales), against PyTorch."""
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(hq * 7 + hkv + (splits or 0) + int8)
    dev = "cuda"
    rows, width, total = 3, 2051, 2400
    blocks = -(-total // BS)
    table = torch.randperm(blocks + 5, generator=g)[:blocks].int().to(dev)
    nblk = blocks + 6
    sel = torch.full((rows, width), -1, dtype=torch.int32)
    for i in range(rows):
        n = (1700, width, 900)[i]
        pick = torch.randperm(total, generator=g)[:n].sort().values.int()
        at = torch.randperm(width, generator=g)[:n].sort().values
        sel[i, at] = pick
    sel = sel.to(dev)
    if int8:
        k_cache = torch.randint(-127, 128, (nblk, BS, hkv, D), generator=g, dtype=torch.int8).to(dev)
        v_cache = torch.randint(-127, 128, (nblk, BS, hkv, D), generator=g, dtype=torch.int8).to(dev)
        k_scale = (torch.rand(nblk, BS, hkv, generator=g) * 0.02 + 0.005).to(dev)
        v_scale = (torch.rand(nblk, BS, hkv, generator=g) * 0.02 + 0.005).to(dev)
        kf = k_cache.float() * k_scale[..., None]
        vf = v_cache.float() * v_scale[..., None]
    else:
        k_cache = torch.randn(nblk, BS, hkv, D, generator=g).half().to(dev)
        v_cache = torch.randn(nblk, BS, hkv, D, generator=g).half().to(dev)
        k_scale = v_scale = None
        kf, vf = k_cache.float(), v_cache.float()
    qa = torch.randn(rows, hq, D, generator=g).half().to(dev)
    qkv = torch.randn(rows, hq * 2 * D, generator=g).half().to(dev)
    qs_t = torch.tensor([0, rows], dtype=torch.int32, device=dev)
    out = kernels.qsa_attention(qa, qkv, k_cache, v_cache, k_scale, v_scale, sel, table[None], qs_t, 1 / 16,
                                splits=splits)
    for i in range(rows):
        p = sel[i][sel[i] >= 0].long()
        s = table[p // BS].long() * BS + p % BS
        kk = kf.view(-1, hkv, D)[s].half().float().repeat_interleave(hq // hkv, dim=1)
        vv = vf.view(-1, hkv, D)[s].half().float().repeat_interleave(hq // hkv, dim=1)
        att = torch.softmax(torch.einsum("hd,khd->hk", qa[i].float(), kk) / 16, dim=-1)
        o = torch.einsum("hk,khd->hd", att, vv).half().float()
        gate = qkv[i].view(hq, 2, D)[:, 1].float()
        want = (o * torch.sigmoid(gate)).reshape(-1)
        torch.testing.assert_close(out[i].float(), want, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_index_prep_matches_norm_rotary_and_key_write(tokens):
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(tokens)
    heads, dim, half = 4, 128, 32
    qk = torch.randn(tokens, (heads + 1) * dim, generator=g).cuda()
    q_norm = (1 + torch.randn(dim, generator=g) * 0.1).cuda()
    ang = torch.rand(tokens, half, generator=g).cuda() * 6
    cos, sin = torch.cos(ang), torch.sin(ang)
    q, raw = kernels.qsa_index_prep(qk, q_norm, cos, sin, heads, 1e-6)
    f = qk[:, : heads * dim].view(tokens, heads, dim)
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6) * q_norm
    want = ops.apply_rotary(f, cos, sin).half()
    torch.testing.assert_close(q.float(), want.float(), rtol=1e-3, atol=1e-3)
    assert torch.equal(raw, qk[:, heads * dim :].half())


@pytest.mark.parametrize("ctx,new", [(0, 300), (1500, 200), (40, 90)])
def test_prefill_selection_matches_the_kernel_scores(ctx, new):
    """qsa_select_prefill (block scores from cuBLAS on the gathered pooled keys) selects the same positions as
    qsa_select (the score kernel on the paged pooled cache)."""
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(ctx + new)
    dev = "cuda"
    heads, dim, c, top, bs = 4, 128, 4, 64, 16
    total = ctx + new
    blocks = -(-total // bs) + 1
    table = torch.randperm(blocks + 3, generator=g)[:blocks].int().to(dev)
    pooled = torch.randn(blocks + 4, bs // c, dim, generator=g).half().to(dev)
    q = torch.randn(new, heads, dim, generator=g).half().to(dev)
    qs = torch.tensor([0, new], dtype=torch.int32, device=dev)
    ctx_t = torch.tensor([ctx], dtype=torch.int32, device=dev)
    want = kernels.qsa_select(q, pooled, table[None], qs, ctx_t, c, top, total // c + 1, bs)
    nb = total // c
    pos = torch.arange(nb, device=dev) * c
    keys = pooled.view(-1, dim)[table[pos // bs].long() * (bs // c) + (pos % bs) // c]
    got = kernels.qsa_select_prefill(q, keys, qs, ctx_t, c, top, 0)
    for i in range(new):
        assert set(got[i].tolist()) == set(want[i].tolist()), i


def test_pool_after_rejected_drafts_uses_the_real_keys():
    """Decode and verify steps of one sequence: a verify block whose rejected drafts complete an indexer block
    pools it from their keys, and the block is pooled again from the real keys when its real last token comes
    (the ring keeps the raw keys of the last 32 positions). The final pooled keys equal one prefill of the real
    tokens."""
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(7)
    dev = "cuda"
    total = 60
    blocks = -(-(total + 8) // BS) + 1
    table = torch.arange(blocks, dtype=torch.int32, device=dev)
    positions = torch.zeros(blocks + 1, BS, 3, dtype=torch.int32, device=dev)
    positions.view(-1, 3)[: blocks * BS] = torch.arange(blocks * BS, device=dev).int()[:, None].expand(-1, 3)
    k_norm = (1 + torch.randn(DIM, generator=g) * 0.1).to(dev)
    inv_freq = (1.0 / (1e7 ** (torch.arange(0, 16, dtype=torch.float64) * 2 / 32))).float().to(dev)
    section = (6, 5, 5)
    real = torch.randn(total + 8, DIM, generator=g).half().to(dev)
    slots_t = torch.tensor([0], dtype=torch.int32, device=dev)

    def step(pooled, ring, keys, ctx):
        n = keys.shape[0]
        kernels.qsa_pool(pooled, keys, ring, slots_t, positions, table[None],
                         torch.tensor([0, n], dtype=torch.int32, device=dev),
                         torch.tensor([ctx], dtype=torch.int32, device=dev), n, k_norm, inv_freq, section, 32, C,
                         BS, 1e-6)

    want = torch.zeros(blocks + 1, BS // C, DIM, dtype=torch.float16, device=dev)
    step(want, torch.zeros(1, 32, DIM, dtype=torch.float16, device=dev), real[:total], 0)
    got = torch.zeros_like(want)
    ring = torch.zeros(1, 32, DIM, dtype=torch.float16, device=dev)
    step(got, ring, real[:21], 0)
    pos = 21
    accepted = (1, 0, 3, 2, 4, 1, 0, 2, 4, 3)
    for n in accepted:
        if pos + 5 > total:
            break
        drafts = real[pos : pos + 5].clone()
        drafts[n + 1 :] = torch.randn(4 - n, DIM, generator=g).half().to(dev)
        step(got, ring, drafts, pos)
        pos += n + 1
    while pos < total:
        step(got, ring, real[pos : pos + 1], pos)
        pos += 1
    nb = total // C
    starts = torch.arange(nb, device=dev) * C
    idx = (table[starts // BS].long() * (BS // C) + (starts % BS) // C)
    assert torch.equal(got.view(-1, DIM)[idx], want.view(-1, DIM)[idx])
