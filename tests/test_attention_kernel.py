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


@pytest.mark.parametrize("lengths, ctx", [([1], [100]), ([1, 1, 1], [0, 37, 250]), ([9, 4], [600, 17])])
def test_prep_and_attention_match_reference(lengths, ctx):
    hq, hkv, d, bs, eps = 6, 2, 256, 16, 1e-6
    qkv, starts, bt, k_cache, v_cache, positions = setup(lengths, ctx, hq, hkv, bs)
    t = qkv.shape[0]
    q_norm = 1.0 + torch.randn(d) * 0.1
    k_norm = 1.0 + torch.randn(d) * 0.1
    inv = 1.0 / (1e7 ** (torch.arange(0, 32).float() * 2 / 64))
    section = (11, 11, 10)
    rk, rv = k_cache.clone(), v_cache.clone()
    qg = qkv[:, : 2 * d * hq].view(t, hq, 2, d)
    q = R.rms_norm(qg[:, :, 0].reshape(-1, d), q_norm, eps)[0].view(t, hq, d)
    k = R.rms_norm(qkv[:, 2 * d * hq : 2 * d * hq + d * hkv].reshape(-1, d), k_norm, eps)[0].view(t, hkv, d)
    v = qkv[:, 2 * d * hq + d * hkv :].reshape(t, hkv, d)
    cos, sin = R.rotary_cos_sin(positions, inv, section)
    q, k = R.apply_rotary(q, cos, sin), R.apply_rotary(k, cos, sin)
    slots = R.kv_slots(starts, ctx, bt.long(), bs)
    R.write_kv(rk, rv, k, v, slots)
    o = R.paged_attention(q, rk, rv, starts, ctx, bt.long(), 1 / math.sqrt(d))
    ref = R.sigmoid_gate(o.reshape(t, hq * d), qg[:, :, 1].reshape(t, hq * d))

    dev = "cuda"
    kk, kv = k_cache.to(dev), v_cache.to(dev)
    qd = kernels.attn_prep(qkv.to(dev), q_norm.to(dev), k_norm.to(dev), positions.to(dev), inv.to(dev), section,
                           slots.int().to(dev), kk, kv, hq, hkv, eps)
    torch.testing.assert_close(qd.cpu().float(), q.float(), rtol=0, atol=2e-3)
    torch.testing.assert_close(kk.cpu().float(), rk.float(), rtol=0, atol=2e-3)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    out = kernels.decode_attention(qd, qkv.to(dev), kk, kv, bt.to(dev), i32(starts), i32(ctx), max(lengths),
                                   bt.shape[1], 1 / math.sqrt(d))
    err = (out.cpu().float() - ref.float()).abs().max().item()
    assert err < 3e-3, err
