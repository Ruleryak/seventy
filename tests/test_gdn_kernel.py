import pytest
import torch

from seventy.ops import kernels
from seventy.ops import reference as R

pytestmark = pytest.mark.gpu


def reference_layer(y, conv_w, conv_state, state, a_log, dt_bias, norm_w, starts, ctx, slots, hk, hv, eps):
    """The same computation with the reference operations, as the model's GatedDeltaNetLayer does it."""
    d = 128
    key, val = hk * d, hv * d
    conv_dim = 2 * key + val
    t = y.shape[0]
    qkv = R.causal_conv1d(y[:, :conv_dim].contiguous(), conv_w, conv_state, starts, ctx, slots)
    z = y[:, conv_dim : conv_dim + val]
    b = y[:, conv_dim + val : conv_dim + val + hv]
    a = y[:, conv_dim + val + hv :]
    q = qkv[:, :key].view(t, hk, d)
    k = qkv[:, key : 2 * key].view(t, hk, d)
    v = qkv[:, 2 * key :].view(t, hv, d)
    g, beta = R.gdn_gates(a, b, a_log, dt_bias)
    o = R.gated_delta_rule(q, k, v, g, beta, state, starts, ctx, slots)
    return R.gated_rms_norm(o.reshape(t * hv, d), z.reshape(t * hv, d), norm_w, eps).view(t, val)


@pytest.mark.parametrize("lengths, ctx, taps", [([1, 1, 1], [5, 0, 9], 4), ([9, 1, 4], [3, 7, 0], 4), ([16], [2], 4),
                                               ([20, 3], [2, 0], 4), ([32], [5], 4), ([9, 2], [4, 0], 3),
                                               ([5, 1], [1, 6], 2)])
def test_gdn_recurrent_matches_reference(lengths, ctx, taps):
    torch.manual_seed(sum(lengths) + sum(ctx))
    hk, hv, d, eps = 2, 6, 128, 1e-6
    conv_dim = 2 * hk * d + hv * d
    width = conv_dim + hv * d + 2 * hv
    t = sum(lengths)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    slots = [2, 0, 3][: len(lengths)]
    y = (torch.randn(t, width) * 0.5).half()
    conv_w = torch.randn(conv_dim, taps) * 0.3
    a_log = torch.randn(hv) * 0.5
    dt_bias = torch.randn(hv) * 0.5
    norm_w = 1.0 + torch.randn(d) * 0.1
    conv_state = (torch.randn(4, conv_dim, taps - 1) * 0.5).half()
    state = torch.randn(4, hv, d, d) * 0.05
    ref_conv, ref_state = conv_state.clone(), state.clone()
    ref = reference_layer(y, conv_w, ref_conv, ref_state, a_log, dt_bias, norm_w, starts, ctx, slots, hk, hv, eps)
    dev = "cuda"
    k_conv, k_state = conv_state.to(dev), state.to(dev)
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    out = kernels.gdn_recurrent(y.to(dev), conv_w.to(dev), k_conv, k_state, a_log.to(dev), dt_bias.to(dev),
                                norm_w.to(dev), i32(starts), i32(ctx), i32(slots), hk, hv, eps)
    err = (out.cpu().float() - ref.float()).abs()
    print(f"\nmax |out diff| {err.max().item():.2e}, mean {err.mean().item():.2e}, max |ref| {ref.float().abs().max().item():.2f}")
    torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=2e-2, atol=2e-2)
    used = sorted(set(slots))
    torch.testing.assert_close(k_state.cpu()[used], ref_state[used], rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(k_conv.cpu()[used].float(), ref_conv[used].float(), rtol=0, atol=0)


def test_verify_then_commit_equals_running_accepted_rows():
    torch.manual_seed(9)
    hk, hv, d, taps, eps = 2, 6, 128, 4, 1e-6
    conv_dim = 2 * hk * d + hv * d
    width = conv_dim + hv * d + 2 * hv
    lengths, ctx, accepted = [8, 8, 8], [10, 3, 0], [3, 8, 1]
    starts = [0, 8, 16, 24]
    y = (torch.randn(24, width) * 0.5).half()
    conv_w = torch.randn(conv_dim, taps) * 0.3
    a_log, dt_bias = torch.randn(hv) * 0.5, torch.randn(hv) * 0.5
    norm_w = 1.0 + torch.randn(d) * 0.1
    conv_state = (torch.randn(3, conv_dim, taps - 1) * 0.5).half()
    state = torch.randn(3, hv, d, d) * 0.05
    dev = "cuda"
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=dev)
    args = (conv_w.to(dev), None, None, a_log.to(dev), dt_bias.to(dev), norm_w.to(dev))
    cs, st = conv_state.to(dev), state.to(dev)
    full = kernels.gdn_recurrent(y.to(dev), conv_w.to(dev), cs, st, a_log.to(dev), dt_bias.to(dev), norm_w.to(dev),
                                 i32(starts), i32(ctx), i32([0, 1, 2]), hk, hv, eps, commit=False)
    assert torch.equal(cs.cpu(), conv_state) and torch.equal(st.cpu(), state)
    kernels.gdn_recurrent(y.to(dev), conv_w.to(dev), cs, st, a_log.to(dev), dt_bias.to(dev), norm_w.to(dev),
                          i32(starts), i32(ctx), i32([0, 1, 2]), hk, hv, eps, commit_lens=i32(accepted))
    rows = torch.cat([y[s : s + n] for s, n in zip(starts, accepted)])
    acc_starts = [0]
    for n in accepted:
        acc_starts.append(acc_starts[-1] + n)
    cs2, st2 = conv_state.to(dev), state.to(dev)
    ref = kernels.gdn_recurrent(rows.to(dev), conv_w.to(dev), cs2, st2, a_log.to(dev), dt_bias.to(dev),
                                norm_w.to(dev), i32(acc_starts), i32(ctx), i32([0, 1, 2]), hk, hv, eps)
    assert torch.equal(st.cpu(), st2.cpu()) and torch.equal(cs.cpu(), cs2.cpu())
    for s, a, n in zip(starts, acc_starts, accepted):
        assert torch.equal(full[s : s + n].cpu(), ref[a : a + n].cpu())
