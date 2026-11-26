"""Rotary frequencies (seventy.rope): the default ones and YaRN against transformers, the YaRN config fields and
the --yarn-factor override, and the attention prep kernel's rope scale."""

import math

import pytest
import torch

from seventy.config import parse_config
from seventy.rope import frequencies


def _raw(rope: dict, max_pos: int = 262144) -> dict:
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text", "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "hidden_size": 2560, "intermediate_size": 9216, "vocab_size": 1000, "rms_norm_eps": 1e-6,
        "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "attn_output_gate": True,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "max_position_embeddings": max_pos,
        "eos_token_id": 2, "dtype": "bfloat16", "rope_parameters": dict(rope),
    }


BASE = {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
        "mrope_interleaved": True}


def test_default_frequencies_are_the_plain_ones():
    t = parse_config(_raw(BASE)).text
    inv, scale = frequencies(t)
    want = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    assert scale == 1.0 and t.max_context == 262144
    torch.testing.assert_close(inv, want.float(), rtol=0, atol=0)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_yarn_matches_transformers(factor):
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from transformers.modeling_rope_utils import _compute_yarn_parameters

    rope = dict(BASE, rope_type="yarn", factor=factor, original_max_position_embeddings=262144)
    t = parse_config(_raw(rope)).text
    inv, scale = frequencies(t)

    class Cfg(SimpleNamespace):
        def standardize_rope_params(self):
            pass

    hf = Cfg(rope_parameters=dict(rope), head_dim=256, hidden_size=2560, num_attention_heads=24,
             max_position_embeddings=262144)
    want, want_scale = _compute_yarn_parameters(hf)
    assert t.rope_type == "yarn" and t.max_context == int(262144 * factor)
    assert scale == pytest.approx(want_scale) and scale == pytest.approx(0.1 * math.log(factor) + 1)
    torch.testing.assert_close(inv, want.float(), rtol=1e-6, atol=0)


def test_yarn_factor_override_extends_a_native_config():
    from seventy.config import with_yarn

    raw = _raw(BASE)
    t = parse_config(raw, yarn_factor=2).text
    assert t.rope_type == "yarn" and t.rope_factor == 2 and t.rope_original_max == 262144
    assert t.max_context == 524288
    t4 = parse_config(with_yarn(with_yarn(raw, 2), 4)).text
    assert t4.rope_original_max == 262144 and t4.max_context == 1048576


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_attn_prep_scales_cos_and_sin():
    from seventy import ops
    from seventy.ops import kernels, kernels_available

    if not kernels_available():
        pytest.skip("needs the kernel library")
    g = torch.Generator().manual_seed(0)
    t, hq, hkv = 5, 4, 1
    qkv = torch.randn(t, hq * 512 + 2 * hkv * 256, generator=g).half().cuda()
    norm = (1 + torch.randn(256, generator=g) * 0.1).cuda()
    pos = torch.arange(100, 100 + t, dtype=torch.int64)[None].expand(3, -1).contiguous().cuda()
    inv = (1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))).float().cuda()
    slots = torch.arange(t, dtype=torch.int32).cuda()
    outs = []
    for scale in (1.0, 1.0693):
        k = torch.zeros(1, 16, hkv, 256, dtype=torch.float16, device="cuda")
        v = torch.zeros_like(k)
        q = kernels.attn_prep(qkv, norm, norm, pos, inv, (11, 11, 10), slots, k, v, hq, hkv, 1e-6, rope_scale=scale)
        outs.append((q.float(), k.view(-1, hkv, 256)[:t].float()))
    (q1, k1), (q2, k2) = outs
    torch.testing.assert_close(q2[..., :64], q1[..., :64] * 1.0693, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(q2[..., 64:], q1[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(k2[..., :64], k1[..., :64] * 1.0693, rtol=2e-3, atol=2e-3)
