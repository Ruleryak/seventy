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


@pytest.mark.parametrize("lengths, ctx", [([1, 1, 1], [5, 0, 9]), ([9, 1, 4], [3, 7, 0]), ([16], [2])])
def test_gdn_recurrent_matches_reference(lengths, ctx):
    torch.manual_seed(sum(lengths) + sum(ctx))
    hk, hv, d, taps, eps = 2, 6, 128, 4, 1e-6
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
