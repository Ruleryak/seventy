import math

import pytest
import torch

from seventy.ops import kernels
from seventy.ops import reference as R

pytestmark = pytest.mark.gpu


def setup(lengths, ctx, hq=6, hkv=2, bs=16, seed=0):
    torch.manual_seed(seed)
    d = 256
    t = sum(lengths)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    width = 2 * hq * d + 2 * hkv * d
    qkv = (torch.randn(t, width) * 0.7).half()
    nblocks = sum(-(-(c + n) // bs) for c, n in zip(ctx, lengths)) + 2
    perm = torch.randperm(nblocks).tolist()
    tables, used = [], 0
    for c, n in zip(ctx, lengths):
        need = -(-(c + n) // bs)
        tables.append(perm[used : used + need])
        used += need
    width_t = max(len(x) for x in tables)
    bt = torch.zeros(len(lengths), width_t, dtype=torch.int32)
    for i, x in enumerate(tables):
        bt[i, : len(x)] = torch.tensor(x)
    k_cache = (torch.randn(nblocks, bs, hkv, d) * 0.7).half()
    v_cache = (torch.randn(nblocks, bs, hkv, d) * 0.7).half()
    pos = []
    for c, n in zip(ctx, lengths):
        pos += list(range(c, c + n))
    positions = torch.tensor([pos, pos, pos])
    return qkv, starts, bt, k_cache, v_cache, positions


@pytest.mark.parametrize("kv", ["fp16", "int8"])
@pytest.mark.parametrize("lengths, ctx, hkv", [([1], [100], 2), ([1, 1, 1], [0, 37, 250], 2), ([9, 4], [600, 17], 2),
                                              ([8, 8, 8, 8], [5000, 3001, 70, 9000], 1), ([8, 3], [12345, 40], 1),
                                              ([8, 3, 1, 5], [1, 63, 64, 120], 1)])
def test_prep_and_attention_match_reference(lengths, ctx, hkv, kv):
    hq, d, bs, eps = 6, 256, 16, 1e-6
    qkv, starts, bt, k_cache, v_cache, positions = setup(lengths, ctx, hq, hkv, bs)
    ks = vs = None
    if kv == "int8":
        k_cache, ks = R.quantize_int8(k_cache)
        v_cache, vs = R.quantize_int8(v_cache)
    t = qkv.shape[0]
    q_norm = 1.0 + torch.randn(d) * 0.1
    k_norm = 1.0 + torch.randn(d) * 0.1
    inv = 1.0 / (1e7 ** (torch.arange(0, 32).float() * 2 / 64))
    section = (11, 11, 10)
    rk, rv = k_cache.clone(), v_cache.clone()
    rks = ks.clone() if ks is not None else None
    rvs = vs.clone() if vs is not None else None
    qg = qkv[:, : 2 * d * hq].view(t, hq, 2, d)
    q = R.rms_norm(qg[:, :, 0].reshape(-1, d), q_norm, eps)[0].view(t, hq, d)
    k = R.rms_norm(qkv[:, 2 * d * hq : 2 * d * hq + d * hkv].reshape(-1, d), k_norm, eps)[0].view(t, hkv, d)
    v = qkv[:, 2 * d * hq + d * hkv :].reshape(t, hkv, d)
    cos, sin = R.rotary_cos_sin(positions, inv, section)
    q, k = R.apply_rotary(q, cos, sin), R.apply_rotary(k, cos, sin)
    slots = R.kv_slots(starts, ctx, bt.long(), bs)
    R.write_kv(rk, rv, k, v, slots, rks, rvs)
    o = R.paged_attention(q, rk, rv, starts, ctx, bt.long(), 1 / math.sqrt(d), rks, rvs)
    ref = R.sigmoid_gate(o.reshape(t, hq * d), qg[:, :, 1].reshape(t, hq * d))

    dev = "cuda"
    kk, kvv = k_cache.to(dev), v_cache.to(dev)
    dks = ks.to(dev) if ks is not None else None
    dvs = vs.to(dev) if vs is not None else None
    qd = kernels.attn_prep(qkv.to(dev), q_norm.to(dev), k_norm.to(dev), positions.to(dev), inv.to(dev), section,
                           slots.int().to(dev), kk, kvv, hq, hkv, eps, dks, dvs)
    torch.testing.assert_close(qd.cpu().float(), q.float(), rtol=0, atol=2e-3)
    if kv == "int8":
        mism = (kk.cpu() != rk).sum().item()
        assert mism <= kk.numel() * 1e-4, mism
        torch.testing.assert_close(dks.cpu(), rks, rtol=1e-3, atol=0)
        assert torch.equal(kvv.cpu(), rv) and torch.equal(dvs.cpu(), rvs)
    else:
        torch.testing.assert_close(kk.cpu().float(), rk.float(), rtol=0, atol=2e-3)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    out = kernels.decode_attention(qd, qkv.to(dev), kk, kvv, bt.to(dev), i32(starts), i32(ctx), max(lengths),
                                   bt.shape[1], 1 / math.sqrt(d), dks, dvs)
    err = (out.cpu().float() - ref.float()).abs().max().item()
    assert err < 3e-3, err


@pytest.mark.parametrize("kv", ["fp16", "int8", "int8-grouped"])
@pytest.mark.parametrize("lengths, ctx, hkv", [([64], [0], 2), ([100, 37], [0, 130], 2), ([200], [70], 2),
                                              ([700, 9, 300], [5000, 0, 63], 1), ([1000], [20000], 1)])
def test_flash_attention_matches_reference(lengths, ctx, hkv, kv):
    hq, d, bs = 6, 256, 16
    qkv, starts, bt, k_cache, v_cache, positions = setup(lengths, ctx, hq, hkv, bs, seed=3)
    t = qkv.shape[0]
    ks = vs = None
    if kv != "fp16":
        k_cache, ks = R.quantize_int8(k_cache)
        v_cache, vs = R.quantize_int8(v_cache)
    q = (torch.randn(t, hq, d) * 0.7).half()
    ref = R.paged_attention(q, k_cache, v_cache, starts, ctx, bt.long(), 1 / math.sqrt(d), ks, vs)
    ref = R.sigmoid_gate(ref.reshape(t, hq * d), qkv[:, : 2 * d * hq].view(t, hq, 2, d)[:, :, 1].reshape(t, hq * d))
    dev = "cuda"
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    out = kernels.flash_attention(q.to(dev), qkv.to(dev), k_cache.to(dev), v_cache.to(dev), bt.to(dev), starts,
                                  i32(starts), i32(ctx), 1 / math.sqrt(d), None if ks is None else ks.to(dev),
                                  None if vs is None else vs.to(dev), ctx if kv == "int8-grouped" else None)
    err = (out.cpu().float() - ref.float()).abs().max().item()
    assert err < 4e-3, err


def test_prep_multimodal_positions_match_reference():
    """attn_prep with distinct temporal, height and width positions (image tokens)."""
    hq, hkv, d, bs, eps = 6, 2, 256, 16, 1e-6
    lengths, ctx = [24], [5]
    qkv, starts, bt, k_cache, v_cache, _ = setup(lengths, ctx, hq, hkv, bs, seed=7)
    t = qkv.shape[0]
    g = torch.Generator().manual_seed(1)
    positions = torch.stack([torch.full((t,), 5), 5 + torch.randint(0, 40, (t,), generator=g),
                             5 + torch.randint(0, 900, (t,), generator=g)])
    q_norm = 1.0 + torch.randn(d) * 0.1
    k_norm = 1.0 + torch.randn(d) * 0.1
    inv = 1.0 / (1e7 ** (torch.arange(0, 32).float() * 2 / 64))
    section = (11, 11, 10)
    qg = qkv[:, : 2 * d * hq].view(t, hq, 2, d)
    q = R.rms_norm(qg[:, :, 0].reshape(-1, d), q_norm, eps)[0].view(t, hq, d)
    k = R.rms_norm(qkv[:, 2 * d * hq : 2 * d * hq + d * hkv].reshape(-1, d), k_norm, eps)[0].view(t, hkv, d)
    cos, sin = R.rotary_cos_sin(positions, inv, section)
    q, k = R.apply_rotary(q, cos, sin), R.apply_rotary(k, cos, sin)
    slots = R.kv_slots(starts, ctx, bt.long(), bs)
    dev = "cuda"
    kk, kvv = k_cache.to(dev), v_cache.to(dev)
    qd = kernels.attn_prep(qkv.to(dev), q_norm.to(dev), k_norm.to(dev), positions.to(dev), inv.to(dev), section,
                           slots.int().to(dev), kk, kvv, hq, hkv, eps, None, None)
    torch.testing.assert_close(qd.cpu().float(), q.float(), rtol=0, atol=2e-3)
    blocks, offs = slots // bs, slots % bs
    torch.testing.assert_close(kk.cpu()[blocks, offs].float(), k.float(), rtol=0, atol=2e-3)
