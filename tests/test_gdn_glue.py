"""The element-wise Gated DeltaNet prefill kernels (csrc/gdn_glue.cu) against seventy.ops.reference."""

import pytest
import torch

from seventy.ops import kernels
from seventy.ops import reference as R

pytestmark = pytest.mark.gpu
DEV = "cuda"


@pytest.mark.parametrize("taps", [4, 2])
def test_conv_gates_and_norm_match_reference(taps):
    torch.manual_seed(taps)
    hk, hv, d = 2, 6, 128
    channels = 2 * hk * d + hv * d
    val = hv * d
    lens = [70, 1, 2, 300, 3]
    ctx = [0, 5, 12, 0, 40]
    slots = [3, 0, 1, 2, 4]
    t = sum(lens)
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    ld = channels + val + 2 * hv
    y = (torch.randn(t, ld, device=DEV) * 2).half()
    w = torch.randn(channels, taps, device=DEV) * 0.5
    state = (torch.randn(6, channels, taps - 1, device=DEV)).half()
    i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEV)

    ref_state = state.clone()
    want = R.causal_conv1d(y[:, :channels].contiguous(), w, ref_state, starts, ctx, slots)
    got_state = state.clone()
    got = kernels.gdn_conv(y, channels, w, got_state, i32(starts), i32(ctx), i32(slots))
    torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=2e-3)
    torch.testing.assert_close(got_state.float(), ref_state.float(), rtol=0, atol=0)
    empty_state = state.clone()
    starts_e = starts[:2] + [starts[1]] + starts[2:]
    got_e = kernels.gdn_conv(y, channels, w, empty_state, i32(starts_e), i32(ctx[:1] + [9] + ctx[1:]),
                             i32(slots[:1] + [5] + slots[1:]))
    assert torch.equal(got_e, got) and torch.equal(empty_state, got_state)

    a_log = torch.randn(hv, device=DEV)
    dt_bias = torch.randn(hv, device=DEV)
    y[:5, channels + val + hv] = 30.0
    g_want, b_want = R.gdn_gates(y[:, channels + val + hv :], y[:, channels + val : channels + val + hv], a_log, dt_bias)
    g, beta = kernels.gdn_gates(y, channels + val + hv, channels + val, hv, a_log, dt_bias)
    torch.testing.assert_close(g, g_want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(beta, b_want, rtol=1e-5, atol=1e-6)

    x = (torch.randn(t, hv, d, device=DEV) * 3).half()
    nw = torch.rand(d, device=DEV) + 0.5
    want_o = R.gated_rms_norm(x.reshape(t * hv, d), y[:, channels : channels + val].reshape(t * hv, d), nw, 1e-6)
    got_o = kernels.gated_rms_norm(x, y, channels, nw, 1e-6)
    torch.testing.assert_close(got_o.view(t * hv, d).float(), want_o.float(), rtol=0, atol=4e-3)
