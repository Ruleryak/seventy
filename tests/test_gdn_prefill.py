import pytest
import torch

from seventy.ops import gdn_prefill
from seventy.ops import reference as R

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("lengths, ctx", [([64], [0]), ([150, 33], [0, 40]), ([300], [17])])
def test_chunked_prefill_matches_reference(lengths, ctx):
    torch.manual_seed(sum(lengths))
    hk, hv, d = 2, 6, 128
    t = sum(lengths)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    slots = [1, 3][: len(lengths)]
    q = torch.randn(t, hk, d).half()
    k = torch.randn(t, hk, d).half()
    v = (torch.randn(t, hv, d) * 0.5).half()
    g = -torch.rand(t, hv) * 0.3
    beta = torch.rand(t, hv)
    state = torch.randn(4, hv, d, d) * 0.05
    ref_state = state.clone()
    ref = R.gated_delta_rule(q, k, v, g, beta, ref_state, starts, ctx, slots)
    dev = "cuda"
    st = state.to(dev)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    out = gdn_prefill.gated_delta_rule(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev), st, starts,
                                       i32(starts), i32(ctx), i32(slots))
    err = (out.cpu().float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    print(f"\nmax |o diff| {err.max().item():.2e} (max |o| {scale:.2f}), mean {err.mean().item():.2e}")
    assert err.max().item() < 2e-3 * scale + 1e-5
    serr = (st.cpu()[slots] - ref_state[slots]).abs().max().item()
    assert serr < 2e-3 * ref_state[slots].abs().max().item() + 1e-5, serr


def _packed(t, hk, hv, d, extra=10):
    """q, k, v as the column blocks of one FP16 buffer (the model's convolution output layout)."""
    buf = torch.randn(t, 2 * hk * d + hv * d + extra).half()
    buf[:, 2 * hk * d :] *= 0.5
    q = buf[:, : hk * d].view(t, hk, d)
    k = buf[:, hk * d : 2 * hk * d].view(t, hk, d)
    v = buf[:, 2 * hk * d : 2 * hk * d + hv * d].view(t, hv, d)
    return buf, q, k, v


@pytest.mark.parametrize("lengths", [[64], [150, 33], [7, 300, 64]])
def test_fused_prepare_matches_torch_prepare(lengths):
    torch.manual_seed(len(lengths) + sum(lengths))
    hk, hv, d = 2, 6, 128
    t = sum(lengths)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    buf, q, k, v = _packed(t, hk, hv, d)
    g = -torch.rand(t, hv) * 0.3
    beta = torch.rand(t, hv)
    dev = "cuda"
    buf = buf.to(dev)
    q, k, v = (x.to(dev) for x in (q, k, v))
    qd = buf[:, : hk * d].view(t, hk, d)
    kd = buf[:, hk * d : 2 * hk * d].view(t, hk, d)
    vd = buf[:, 2 * hk * d : 2 * hk * d + hv * d].view(t, hv, d)
    gd, bd = g.to(dev), beta.to(dev)
    assert gdn_prefill._fused_layout(qd, kd, vd, gd, bd)
    want = gdn_prefill.prepare(q.contiguous(), k.contiguous(), v.contiguous(), gd, bd, starts)
    got = gdn_prefill.prepare_fused(qd, kd, vd, gd, bd, starts)
    assert got["chunks"] == want["chunks"] and torch.equal(got["chunk_start"], want["chunk_start"])
    for name in ("w", "u", "gq", "p", "klt", "eg"):
        a, b = got[name].float(), want[name].float()
        err = (a - b).abs().max().item()
        print(f"\n{name}: max |diff| {err:.2e}, max |ref| {b.abs().max().item():.2f}")
        assert err <= 2e-3 * b.abs().max().item() + 1e-4, name


@pytest.mark.parametrize("lengths, ctx", [([150, 33], [0, 40]), ([300], [17])])
def test_fused_chunked_prefill_matches_reference(lengths, ctx):
    torch.manual_seed(sum(lengths) + 1)
    hk, hv, d = 2, 6, 128
    t = sum(lengths)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    slots = [1, 3][: len(lengths)]
    buf, q, k, v = _packed(t, hk, hv, d)
    g = -torch.rand(t, hv) * 0.3
    beta = torch.rand(t, hv)
    state = torch.randn(4, hv, d, d) * 0.05
    ref_state = state.clone()
    ref = R.gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(), g, beta, ref_state, starts, ctx, slots)
    dev = "cuda"
    buf = buf.to(dev)
    qd = buf[:, : hk * d].view(t, hk, d)
    kd = buf[:, hk * d : 2 * hk * d].view(t, hk, d)
    vd = buf[:, 2 * hk * d : 2 * hk * d + hv * d].view(t, hv, d)
    st = state.to(dev)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    out = gdn_prefill.gated_delta_rule(qd, kd, vd, g.to(dev), beta.to(dev), st, starts, i32(starts), i32(ctx),
                                       i32(slots))
    err = (out.cpu().float() - ref.float()).abs()
    scale = ref.float().abs().max().item()
    print(f"\nmax |o diff| {err.max().item():.2e} (max |o| {scale:.2f})")
    assert err.max().item() < 2e-3 * scale + 1e-5
    serr = (st.cpu()[slots] - ref_state[slots]).abs().max().item()
    assert serr < 2e-3 * ref_state[slots].abs().max().item() + 1e-5, serr
