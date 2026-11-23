import copy

import pytest

from seventy.config import UnsupportedModel, load_config, parse_config


def small_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_id": 9,
        "text_config": {
            "model_type": "qwen3_5_text",
            "attn_output_gate": True,
            "attention_bias": False,
            "eos_token_id": 2,
            "head_dim": 32,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 16,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
            "vocab_size": 100,
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
        },
    }


def test_small_config_parses():
    cfg = parse_config(small_config())
    t = cfg.text
    assert t.rotary_dim == 8
    assert t.linear_layers == (0, 1, 2)
    assert t.attention_layers == (3,)
    assert t.conv_dim == 2 * 2 * 16 + 4 * 16
    assert cfg.quant.method == "bf16"
    assert cfg.vision is not None and cfg.vision.depth == 2


def test_generation_config_adds_eos():
    cfg = parse_config(small_config(), generation={"eos_token_id": [2, 7]})
    assert cfg.text.eos_token_ids == (2, 7)


@pytest.mark.parametrize(
    "change, fragment",
    [
        (lambda c: c.update(architectures=["LlamaForCausalLM"]), "architecture"),
        (lambda c: c["text_config"].update(attn_output_gate=False), "output gate"),
        (lambda c: c["text_config"].update(layer_types=["linear_attention"] * 3 + ["sliding"]), "unknown layer"),
        (lambda c: c["text_config"]["rope_parameters"].update(rope_type="yarn"), "rope type"),
        (lambda c: c["text_config"]["rope_parameters"].update(mrope_section=[2, 2, 2]), "mrope_section"),
        (lambda c: c["text_config"]["rope_parameters"].update(mrope_interleaved=False), "interleaved"),
        (lambda c: c["text_config"].update(linear_num_value_heads=3), "multiple"),
        (lambda c: c.update(dtype="float32"), "bfloat16 or float16"),
        (lambda c: c.pop("vision_config"), "vision_config"),
    ],
)
def test_unsupported(change, fragment):
    raw = copy.deepcopy(small_config())
    change(raw)
    with pytest.raises(UnsupportedModel) as info:
        parse_config(raw)
    assert any(fragment in r for r in info.value.reasons), info.value.reasons


def test_all_reasons_reported():
    raw = small_config()
    raw["architectures"] = ["Other"]
    raw["text_config"]["hidden_act"] = "gelu"
    with pytest.raises(UnsupportedModel) as info:
        parse_config(raw)
    assert len(info.value.reasons) == 2


def nvfp4_quant(group_size=16):
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "symmetric": True,
                    "group_size": group_size,
                    "scale_dtype": "torch.float8_e4m3fn",
                }
            }
        },
        "ignore": ["lm_head", "re:.*visual.*"],
    }


def test_nvfp4_config():
    raw = small_config()
    raw["quantization_config"] = nvfp4_quant()
    cfg = parse_config(raw)
    assert cfg.quant.method == "nvfp4" and cfg.quant.group_size == 16


def test_nvfp4_wrong_group():
    raw = small_config()
    raw["quantization_config"] = nvfp4_quant(group_size=32)
    with pytest.raises(UnsupportedModel, match="group size 16"):
        parse_config(raw)


@pytest.mark.model
def test_real_model_config(model_path):
    cfg = load_config(model_path)
    t = cfg.text
    assert cfg.quant.method == "nvfp4"
    assert t.num_layers == 64 and len(t.attention_layers) == 16
    assert t.attention_layers[:2] == (3, 7)
    assert t.rotary_dim == 64 and t.mrope_section == (11, 11, 10)
    assert t.conv_dim == 10240
    assert 248044 in t.eos_token_ids


def test_dtype_in_the_text_config_counts():
    raw = small_config()
    del raw["dtype"]
    raw["text_config"]["dtype"] = "bfloat16"
    assert parse_config(raw).quant.method == "bf16"


def test_kernel_problems_name_the_head_sizes():
    from seventy.config import kernel_problems

    problems = kernel_problems(parse_config(small_config()))
    assert any("attention head size 32" in p for p in problems)
    assert any("Gated DeltaNet head sizes 16/16" in p for p in problems)
    raw = small_config()
    raw["text_config"].update(head_dim=256, linear_key_head_dim=128, linear_value_head_dim=128,
                              rope_parameters=dict(raw["text_config"]["rope_parameters"], mrope_section=[11, 11, 10]))
    assert kernel_problems(parse_config(raw)) == []
