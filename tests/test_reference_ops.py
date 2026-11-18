import math

import pytest
import torch
import torch.nn.functional as F

from seventy.ops import reference as R


def gdn_inputs(lengths, hk=2, hv=4, dk=16, dv=8, seed=0):
    torch.manual_seed(seed)
    t = sum(lengths)
    q = torch.randn(t, hk, dk)
    k = torch.randn(t, hk, dk)
    v = torch.randn(t, hv, dv)
    g = -torch.rand(t, hv) * 0.5
    beta = torch.rand(t, hv)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    return q, k, v, g, beta, starts


@pytest.mark.parametrize("chunk", [4, 64])
@pytest.mark.parametrize("with_context", [False, True])
def test_chunked_matches_recurrent(chunk, with_context):
    lengths = [5, 13, 1, 9]
    q, k, v, g, beta, starts = gdn_inputs(lengths)
    ctx = [7, 0, 3, 2] if with_context else [0] * 4
    slots = [3, 0, 2, 1]
    state0 = torch.randn(4, 4, 16, 8)
    s1, s2 = state0.clone(), state0.clone()
    o1 = R.gated_delta_rule_recurrent(q, k, v, g, beta, s1, starts, ctx, slots)
    o2 = R.gated_delta_rule(q, k, v, g, beta, s2, starts, ctx, slots, chunk=chunk)
    torch.testing.assert_close(o2, o1, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s2, s1, rtol=1e-4, atol=1e-4)


def test_delta_rule_split_equals_whole():
    q, k, v, g, beta, _ = gdn_inputs([20])
    whole = torch.zeros(1, 4, 16, 8)
    o_whole = R.gated_delta_rule(q, k, v, g, beta, whole, [0, 20], [0], [0])
    part = torch.zeros(1, 4, 16, 8)
    o_a = R.gated_delta_rule(q[:12], k[:12], v[:12], g[:12], beta[:12], part, [0, 12], [0], [0])
    o_b = R.gated_delta_rule(q[12:], k[12:], v[12:], g[12:], beta[12:], part, [0, 8], [12], [0])
    torch.testing.assert_close(torch.cat((o_a, o_b)), o_whole, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(part, whole, rtol=1e-4, atol=1e-5)


def test_delta_rule_matches_transformers():
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    q, k, v, g, beta, starts = gdn_inputs([37], hk=2, hv=4)
    state = torch.zeros(1, 4, 16, 8)
    ours = R.gated_delta_rule(q, k, v, g, beta, state, starts, [0], [0])
    rep = 2
    hf_out, hf_state = hf.torch_chunk_gated_delta_rule(
        q.repeat_interleave(rep, 1)[None], k.repeat_interleave(rep, 1)[None], v[None], g[None], beta[None],
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(ours, hf_out[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(state[0], hf_state[0], rtol=1e-4, atol=1e-5)


def test_conv_split_equals_whole():
    torch.manual_seed(1)
    x = torch.randn(10, 6)
    w = torch.randn(6, 4)
    st = torch.zeros(1, 6, 3)
    whole = R.causal_conv1d(x, w, st, [0, 10], [0], [0])
    st2 = torch.zeros(1, 6, 3)
    a = R.causal_conv1d(x[:2], w, st2, [0, 2], [0], [0])
    b = R.causal_conv1d(x[2:], w, st2, [0, 8], [2], [0])
    torch.testing.assert_close(torch.cat((a, b)), whole)
    torch.testing.assert_close(st2, st)
    expected = F.silu(F.conv1d(x.T[None], w[:, None, :], padding=3, groups=6)[0, :, :10]).T
    torch.testing.assert_close(whole, expected)


def test_conv_matches_transformers():
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.manual_seed(2)
    x = torch.randn(9, 8)
    w = torch.randn(8, 4)
    ours = R.causal_conv1d(x, w, torch.zeros(1, 8, 3), [0, 9], [0], [0])
    theirs = hf.causal_conv1d_fn(x.T[None], w, None, activation="silu")[0].T
    torch.testing.assert_close(ours, theirs)


def test_paged_attention_matches_dense():
    torch.manual_seed(3)
    hq, hkv, d, bs = 6, 2, 16, 4
    ctx, n = 9, 5
    total = ctx + n
    q = torch.randn(n, hq, d)
    k_all = torch.randn(total, hkv, d)
    v_all = torch.randn(total, hkv, d)
    blocks = torch.tensor([[7, 2, 5, 0]])
    k_cache = torch.zeros(8, bs, hkv, d)
    v_cache = torch.zeros(8, bs, hkv, d)
    slots = R.kv_slots([0, total], [0], blocks, bs)
    R.write_kv(k_cache, v_cache, k_all, v_all, slots)
    out = R.paged_attention(q, k_cache, v_cache, [0, n], [ctx], blocks, 1 / math.sqrt(d))
    kk = k_all.repeat_interleave(3, 1).transpose(0, 1)
    vv = v_all.repeat_interleave(3, 1).transpose(0, 1)
    mask = torch.ones(n, total, dtype=torch.bool).tril(diagonal=ctx)
    ref = F.scaled_dot_product_attention(q.transpose(0, 1), kk, vv, attn_mask=mask).transpose(0, 1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rotary_text_positions_are_standard_rope():
    half, theta = 32, 1e7
    inv = 1.0 / (theta ** (torch.arange(0, 2 * half, 2).float() / (2 * half)))
    pos = torch.arange(11)
    cos, sin = R.rotary_cos_sin(pos.expand(3, -1), inv, (11, 11, 10))
    torch.testing.assert_close(cos, torch.cos(pos[:, None].float() * inv))
    x = torch.randn(11, 2, 256)
    y = R.apply_rotary(x, cos, sin)
    assert torch.equal(y[..., 64:], x[..., 64:])
    a, b = x[..., :32], x[..., 32:64]
    c, s = cos[:, None], sin[:, None]
    torch.testing.assert_close(y[..., :32], a * c - b * s)
    torch.testing.assert_close(y[..., 32:64], b * c + a * s)


def test_rotary_matches_transformers_mrope():
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(head_dim=256, hidden_size=512, num_attention_heads=2,
                            rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                             "mrope_section": [11, 11, 10], "mrope_interleaved": True})
    rope = hf.Qwen3_5TextRotaryEmbedding(cfg)
    torch.manual_seed(4)
    pos = torch.randint(0, 500, (3, 1, 7))
    hf_cos, hf_sin = rope(torch.zeros(1, dtype=torch.float32), pos)
    cos, sin = R.rotary_cos_sin(pos[:, 0], rope.inv_freq, (11, 11, 10))
    torch.testing.assert_close(torch.cat((cos, cos), -1), hf_cos[0])
    torch.testing.assert_close(torch.cat((sin, sin), -1), hf_sin[0])


def test_gated_rms_norm_and_gates():
    torch.manual_seed(5)
    x, z = torch.randn(4, 8), torch.randn(4, 8)
    w = torch.rand(8)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w * F.silu(z)
    torch.testing.assert_close(R.gated_rms_norm(x, z, w, 1e-6), ref)
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    a_log, dt = torch.randn(4), torch.randn(4)
    g, beta = R.gdn_gates(a, b, a_log, dt)
    torch.testing.assert_close(g, -a_log.exp() * F.softplus(a + dt))
    torch.testing.assert_close(beta, torch.sigmoid(b))
