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
