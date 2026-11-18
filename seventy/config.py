"""Model configuration: read a checkpoint's config.json, check that seventy can serve it, and derive sizes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
TEXT_MODEL_TYPES = ("qwen3_5_text", "qwen3_5")
LAYER_TYPES = ("linear_attention", "full_attention")


class UnsupportedModel(ValueError):
    """The checkpoint is not one seventy can serve. `reasons` lists every problem found."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    layer_types: tuple[str, ...]
    vocab_size: int
    rms_norm_eps: float
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    mrope_section: tuple[int, int, int]
    gdn_k_heads: int
    gdn_v_heads: int
    gdn_k_dim: int
    gdn_v_dim: int
    conv_kernel: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]

    @property
    def linear_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @property
    def gdn_key_size(self) -> int:
        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def gdn_value_size(self) -> int:
        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.gdn_key_size + self.gdn_value_size


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    hidden_act: str


@dataclass(frozen=True)
class QuantConfig:
    method: str
    group_size: int = 0
    ignore: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    path: Path | None
    architecture: str
    text: TextConfig
    vision: VisionConfig | None
    quant: QuantConfig
    image_token_id: int | None
    video_token_id: int | None
    vision_start_token_id: int | None
    vision_end_token_id: int | None


def load_config(path: str | Path) -> ModelConfig:
    """Read config.json (and generation_config.json, if present) from a checkpoint folder."""
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    generation = None
    if (path / "generation_config.json").is_file():
        generation = json.loads((path / "generation_config.json").read_text())
    return parse_config(raw, path=path, generation=generation)


def parse_config(raw: dict, path: Path | None = None, generation: dict | None = None) -> ModelConfig:
    reasons: list[str] = []
    architectures = raw.get("architectures") or []
    architecture = architectures[0] if architectures else ""
    if architecture not in ARCHITECTURES:
        reasons.append(f"architecture {architecture or '(none)'} is not supported; supported: {', '.join(ARCHITECTURES)}")
    text_raw = raw.get("text_config", raw)
    text = _parse_text(text_raw, generation, reasons)
    vision = None
    if architecture == "Qwen3_5ForConditionalGeneration" and not raw.get("language_model_only", False):
        vision = _parse_vision(raw.get("vision_config"), reasons)
    quant = _parse_quant(raw.get("quantization_config"), raw.get("dtype") or raw.get("torch_dtype"), reasons)
    if reasons:
        raise UnsupportedModel(reasons)
    return ModelConfig(
        path=path,
        architecture=architecture,
        text=text,
        vision=vision,
        quant=quant,
        image_token_id=raw.get("image_token_id"),
        video_token_id=raw.get("video_token_id"),
        vision_start_token_id=raw.get("vision_start_token_id"),
        vision_end_token_id=raw.get("vision_end_token_id"),
    )


def _require(raw: dict, key: str, reasons: list[str], kind=int):
    value = raw.get(key)
    if value is None:
        reasons.append(f"text config has no {key}")
        return kind()
    return kind(value)


def _parse_text(raw: dict, generation: dict | None, reasons: list[str]) -> TextConfig | None:
    model_type = raw.get("model_type")
    if model_type not in TEXT_MODEL_TYPES:
        reasons.append(f"text model_type {model_type!r} is not supported")
    num_layers = _require(raw, "num_hidden_layers", reasons)
    layer_types = tuple(raw.get("layer_types") or ())
    if len(layer_types) != num_layers:
        reasons.append(f"layer_types has {len(layer_types)} entries for {num_layers} layers")
    unknown = sorted(set(layer_types) - set(LAYER_TYPES))
    if unknown:
        reasons.append(f"unknown layer types: {', '.join(unknown)}")
    if raw.get("hidden_act", "silu") != "silu":
        reasons.append(f"hidden_act {raw.get('hidden_act')!r} is not supported (only silu)")
    if not raw.get("attn_output_gate", False):
        reasons.append("attention without an output gate is not supported")
    if raw.get("attention_bias", False):
        reasons.append("attention bias is not supported")

    head_dim = _require(raw, "head_dim", reasons)
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default":
        reasons.append(f"rope type {rope_type!r} is not supported")
    factor = rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 1.0))
    rotary_dim = int(round(head_dim * factor))
    if rotary_dim <= 0 or rotary_dim % 2 or abs(rotary_dim - head_dim * factor) > 1e-6 or rotary_dim > head_dim:
        reasons.append(f"partial_rotary_factor {factor} does not give an even rotary size for head_dim {head_dim}")
    section = tuple(rope.get("mrope_section") or (rotary_dim // 2, 0, 0))
    if len(section) != 3 or sum(section) != rotary_dim // 2:
        reasons.append(f"mrope_section {list(section)} does not cover {rotary_dim // 2} rotary frequencies")
        section = (rotary_dim // 2, 0, 0)
    if section[1] or section[2]:
        if not rope.get("mrope_interleaved", False):
            reasons.append("chunked (non-interleaved) mrope is not supported")
    theta = rope.get("rope_theta", raw.get("rope_theta"))
    if theta is None:
        reasons.append("text config has no rope_theta")
        theta = 0.0

    num_heads = _require(raw, "num_attention_heads", reasons)
    num_kv_heads = _require(raw, "num_key_value_heads", reasons)
    if num_kv_heads and num_heads % num_kv_heads:
        reasons.append(f"{num_heads} attention heads are not a multiple of {num_kv_heads} KV heads")
    gdn_k_heads = _require(raw, "linear_num_key_heads", reasons)
    gdn_v_heads = _require(raw, "linear_num_value_heads", reasons)
    if gdn_k_heads and gdn_v_heads % gdn_k_heads:
        reasons.append(f"{gdn_v_heads} linear value heads are not a multiple of {gdn_k_heads} key heads")
    conv_kernel = _require(raw, "linear_conv_kernel_dim", reasons)
    if not 2 <= conv_kernel <= 4:
        reasons.append(f"linear_conv_kernel_dim {conv_kernel} is not supported (2 to 4)")
    ssm_dtype = raw.get("mamba_ssm_dtype", "float32")
    if ssm_dtype != "float32":
        reasons.append(f"mamba_ssm_dtype {ssm_dtype!r} is not supported (only float32)")
    if raw.get("mtp_num_hidden_layers", 0) and raw.get("mtp_use_dedicated_embeddings", False):
        reasons.append("MTP with dedicated embeddings is not supported")

    eos = _token_ids(raw.get("eos_token_id"))
    if generation:
        eos = tuple(dict.fromkeys(eos + _token_ids(generation.get("eos_token_id"))))

    return TextConfig(
        hidden_size=_require(raw, "hidden_size", reasons),
        intermediate_size=_require(raw, "intermediate_size", reasons),
        num_layers=num_layers,
        layer_types=layer_types,
        vocab_size=_require(raw, "vocab_size", reasons),
        rms_norm_eps=_require(raw, "rms_norm_eps", reasons, float),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope_theta=float(theta),
        mrope_section=section,
        gdn_k_heads=gdn_k_heads,
        gdn_v_heads=gdn_v_heads,
        gdn_k_dim=_require(raw, "linear_key_head_dim", reasons),
        gdn_v_dim=_require(raw, "linear_value_head_dim", reasons),
        conv_kernel=conv_kernel,
        max_position_embeddings=_require(raw, "max_position_embeddings", reasons),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        eos_token_ids=eos,
    )


def _token_ids(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def _parse_vision(raw: dict | None, reasons: list[str]) -> VisionConfig | None:
    if not raw:
        reasons.append("the model has no vision_config")
        return None
    if raw.get("deepstack_visual_indexes"):
        reasons.append("deepstack vision layers are not supported")
    try:
        return VisionConfig(
            depth=int(raw["depth"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_heads=int(raw["num_heads"]),
            in_channels=int(raw.get("in_channels", 3)),
            patch_size=int(raw["patch_size"]),
            temporal_patch_size=int(raw["temporal_patch_size"]),
            spatial_merge_size=int(raw["spatial_merge_size"]),
            out_hidden_size=int(raw["out_hidden_size"]),
            num_position_embeddings=int(raw["num_position_embeddings"]),
            hidden_act=str(raw.get("hidden_act", "gelu_pytorch_tanh")),
        )
    except KeyError as missing:
        reasons.append(f"vision_config has no {missing.args[0]}")
        return None


def _parse_quant(raw: dict | None, dtype: str | None, reasons: list[str]) -> QuantConfig:
    if raw is None:
        if dtype not in ("bfloat16", "float16"):
            reasons.append(f"unquantized weights must be bfloat16 or float16, not {dtype!r}")
        return QuantConfig(method="bf16" if dtype == "bfloat16" else "fp16")
    method = raw.get("quant_method")
    fmt = raw.get("format")
    if method != "compressed-tensors" or fmt != "nvfp4-pack-quantized":
        reasons.append(f"quantization {method}/{fmt} is not supported (only compressed-tensors nvfp4-pack-quantized)")
        return QuantConfig(method="unsupported")
    group_size = 0
    for name, group in (raw.get("config_groups") or {}).items():
        weights = group.get("weights") or {}
        problems = []
        if weights.get("num_bits") != 4 or weights.get("type") != "float":
            problems.append("4-bit float weights")
        if weights.get("strategy") != "tensor_group" or not weights.get("symmetric", False):
            problems.append("symmetric tensor_group strategy")
        if weights.get("scale_dtype") not in (None, "torch.float8_e4m3fn"):
            problems.append("FP8 E4M3 scales")
        size = weights.get("group_size")
        if size != 16:
            problems.append("group size 16")
        if problems:
            reasons.append(f"quantization group {name} needs {', '.join(problems)}")
        group_size = size or group_size
    if not group_size:
        reasons.append("quantization config has no weight groups")
    ignore = tuple(raw.get("ignore") or ())
    for pattern in ignore:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:])
            except re.error:
                reasons.append(f"quantization ignore pattern {pattern!r} is not a valid regular expression")
    return QuantConfig(method="nvfp4", group_size=group_size, ignore=ignore)
