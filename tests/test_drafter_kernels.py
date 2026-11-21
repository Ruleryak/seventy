"""The fused DFlash2 drafter kernels (csrc/drafter.cu) against the drafter's PyTorch reference code."""

import math
import types

import pytest
import torch

from seventy.model.dflash2 import Drafter
from seventy.ops import kernels
from seventy.ops import reference as R

pytestmark = pytest.mark.gpu
DEV = "cuda"


def ref_drafter(**cfg):
    """A Drafter object with only what the reference helpers read (no checkpoint)."""
    d = Drafter.__new__(Drafter)
    d.cfg = types.SimpleNamespace(eps=1e-6, block=8, group_size=16, taps=2, window=cfg.get("window", 2048),
                                  ring=cfg.get("ring", 2064), head_dim=128)
    d.groups = cfg.get("groups", 320)
    d.dtype = torch.float16
    d.fused = False
    d.par = types.SimpleNamespace(all_reduce=lambda x: x, size=1)
    return d


@pytest.mark.parametrize("fp32, with_res", [(True, False), (True, True), (False, True)])
def test_drafter_norm(fp32, with_res):
    torch.manual_seed(0)
    x = torch.randn(13, 5120, device=DEV) * 30
    x = x if fp32 else x.half()
    res = torch.randn(13, 5120, device=DEV) * 500 if with_res else None
    w = torch.rand(5120, device=DEV) + 0.5
    ref_out, ref_res = ref_drafter()._norm(x, w, res)
    out, new_res = kernels.drafter_norm(x, w, 1e-6, res)
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=2e-3, atol=2e-3)
    if with_res:
        torch.testing.assert_close(new_res, ref_res, rtol=0, atol=0)
    else:
        assert new_res is None


@pytest.mark.parametrize("out_fp32", [False, True])
def test_drafter_conv(out_fp32):
    torch.manual_seed(1)
    rows, groups, gs = 24, 320, 16
    x = torch.randn(rows, groups * gs, device=DEV).half()
    base = torch.randn(2, 2, groups * gs, device=DEV)
    coef = torch.randn(rows, 2, 2, groups, device=DEV)
    d = ref_drafter()
    for side in (0, 1):
        ref = d._conv(x, base[side], coef[:, side], torch.float32 if out_fp32 else None)
        out = kernels.drafter_conv(x, base[side], coef[:, side], gs, 8, out_fp32)
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


def test_drafter_qk_and_attention():
    torch.manual_seed(2)
    b, blk, heads, kv, d, ring, window = 3, 8, 8, 2, 128, 2064, 2048
    slots_n = 5
    k_ring = (torch.randn(slots_n, ring, kv, d, device=DEV) * 0.5).half()
    v_ring = (torch.randn(slots_n, ring, kv, d, device=DEV) * 0.5).half()
    starts = torch.tensor([0, 37, 3000], device=DEV)
    slots = torch.tensor([1, 4, 2], device=DEV)
    positions = (starts[:, None] + torch.arange(blk, device=DEV)).reshape(-1)
    rows = b * blk
    q = torch.randn(rows, heads * d, device=DEV) * 3
    kvp = torch.randn(rows, 2 * kv * d, device=DEV) * 3
    q_norm = torch.rand(d, device=DEV) + 0.5
    k_norm = torch.rand(d, device=DEV) + 0.5
    inv = (1.0 / (1e6 ** (torch.arange(0, 64, dtype=torch.float64) * 2 / 128))).float().to(DEV)
    ref = ref_drafter()
    ref.inv_freq = inv
    ref.k_ring, ref.v_ring = k_ring.clone()[None], v_ring.clone()[None]
    layer = types.SimpleNamespace(heads=heads, kv=kv, q=None, kv_w=None, o=None, q_norm=q_norm, k_norm=k_norm)
    captured = {}

    def fake_mm(x, w, row_scale=None):
        if w == "q":
            return q
        if w == "kv":
            return kvp
        captured["o"] = x
        return x.float()

    layer.q, layer.kv_w, layer.o = "q", "kv", "o"
    ref._mm = fake_mm
    ref._mm_scaled = lambda x, w: fake_mm(x, w)
    ref._attention(layer, 0, torch.zeros(rows, 1, device=DEV), positions, slots, starts + blk)
    want = captured["o"]

    kr, vr = k_ring.clone(), v_ring.clone()
    q16 = kernels.drafter_qk(q, kvp, q_norm, k_norm, positions, slots.repeat_interleave(blk), inv, kr, vr, heads, kv,
                             1e-6)
    idx = positions % ring
    rs = slots.repeat_interleave(blk)
    torch.testing.assert_close(kr[rs, idx].float(), ref.k_ring[0][rs, idx].float(), rtol=0, atol=4e-3)
    torch.testing.assert_close(vr[rs, idx].float(), ref.v_ring[0][rs, idx].float(), rtol=0, atol=0)
    assert torch.equal(kr[0], k_ring[0]) and torch.equal(kr[3], k_ring[3])
    out = kernels.drafter_attention(q16, kr, vr, slots, starts, blk, window, 1.0 / math.sqrt(d))
    err = (out.float() - want.float()).abs().max().item()
    assert err < 2e-2 * max(1.0, want.float().abs().max().item()), err


def test_silu_mul_scale_and_row_scale_gemm():
    torch.manual_seed(3)
    rows, inter, n = 10, 4352, 640
    gu = torch.randn(rows, 2 * inter, device=DEV) * 20
    gu[3] *= 400
    out, scale = kernels.silu_mul_scale(gu)
    gate, up = gu.chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up
    assert scale[3] > 1 and scale[0] == 1
    torch.testing.assert_close(out.float() * scale[:, None], ref, rtol=2e-3, atol=2e-3)
    w = (torch.randn(n, inter, device=DEV) * 0.02).half()
    packed = kernels.DenseWeight(w)
    got = kernels.dense_gemm(out, packed, row_scale=scale)
    want = (out.float() @ w.float().T) * scale[:, None]
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-2 * want.abs().max().item() / 100)


@pytest.mark.parametrize("b, n, k, rank, sampled", [(1, 7, 16, 256, False), (5, 7, 16, 256, True), (3, 4, 8, 64, True),
                                                    (8, 7, 16, 256, False)])
def test_drafter_walk_matches_reference(b, n, k, rank, sampled):
    from types import SimpleNamespace

    from seventy.model.dflash2 import Drafter

    torch.manual_seed(b * 100 + k)
    vocab = 3000
    d = SimpleNamespace(cfg=SimpleNamespace(top_k=k), sel_hidden=None, pred=(torch.randn(vocab, rank) * 0.3).half().cuda(),
                        succ=(torch.randn(vocab, rank) * 0.3).half().cuda())
    anchors = torch.randint(0, vocab, (b,)).cuda()
    values = (torch.randn(b, n, k) * 3).cuda()
    ids = torch.stack([torch.randperm(vocab)[:k] for _ in range(b * n)]).view(b, n, k).cuda()
    hid = torch.randn(b, n, rank).cuda()
    temps = torch.rand(b).cuda() + 0.3 if sampled else torch.zeros(b).cuda()
    if sampled:
        temps[0] = 0.0
    gumbel = -torch.log(-torch.log(torch.rand(b, n, k).clamp(1e-6, 1 - 1e-6))).cuda() if sampled else None
    d._mm = lambda h, w: h
    import os

    os.environ["SEVENTY_DRAFTER_WALK"] = "torch"
    ref = Drafter.walk(d, anchors.view(b, 1), values, ids, hid, temps, gumbel)
    del os.environ["SEVENTY_DRAFTER_WALK"]
    got = Drafter.walk(d, anchors.view(b, 1), values, ids, hid, temps, gumbel)
    assert torch.equal(got.tokens, ref.tokens)
    assert torch.allclose(got.probs, ref.probs, atol=2e-5, rtol=1e-4)
