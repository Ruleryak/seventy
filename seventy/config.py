"""Model configuration: read a checkpoint's config.json, check that seventy can serve it, and derive sizes."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5MoeForConditionalGeneration",
                 "Qwen3_5MoeForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
MULTIMODAL_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
                            "Qwen4ExpForConditionalGeneration")
TEXT_MODEL_TYPES = ("qwen3_5_text", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_moe", "qwen4_exp_text", "qwen4_exp")
MODELOPT_ALGOS = ("NVFP4", "W4A16_NVFP4", "FP8", "MXFP8")
"""ModelOpt formats seventy reads: NVFP4 weights (the activation part of NVFP4 W4A4 is not used; activations stay
FP16), FP8 E4M3 weights with one scale per tensor, and MXFP8 (E4M3 weights with a power-of-two scale per 32
columns; loaded as FP16)."""
LAYER_TYPES = ("linear_attention", "full_attention", "indexed_attention")
ATTENTION_HEAD_DIM = 256
GDN_HEAD_DIM = 128


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
    num_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    exp: "ExpConfig | None" = None
    rope_type: str = "default"
    rope_factor: float = 1.0
    rope_original_max: int = 0
    rope_attention_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_truncate: bool = True

    @property
    def max_context(self) -> int:
        """The longest context: max_position_embeddings, or with YaRN the original length times the factor if
        that is longer."""
        if self.rope_type == "yarn":
            return max(self.max_position_embeddings, int(self.rope_original_max * self.rope_factor))
        return self.max_position_embeddings

    @property
    def is_moe(self) -> bool:
        """Whether the MLP of every layer is a mixture of experts (with a shared expert) instead of a dense MLP."""
        return self.num_experts > 0

    @property
    def linear_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t in ("full_attention", "indexed_attention"))

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
class ExpConfig:
    """The parts of Qwen4-Exp (Qwen3.8 Flash-Next) text models beyond Qwen3.5-MoE.

    - Hyper-connections: hc_count residual streams of hidden_size; before each attention and MLP block a gated
      low-rank mix (rank hc_lowrank) of the streams is the block input, and the block output is added back to every
      stream with its own weight.
    - Per-layer n-gram embeddings (PLE) at ple_layers: hashed 2- to ngram_size-grams of the token history pick rows
      of an embedding table (heads_per_ngram heads per n-gram size), which a gated projection and a dilated causal
      convolution add to the streams.
    - Qwen Sparse Attention: an indexer (indexer_heads query heads, one key head of indexer_head_dim) scores blocks
      of indexer_compress keys, and each query attends to the indexer_budget tokens of its best blocks plus the
      tokens of the unfinished last block."""

    hc_count: int
    hc_lowrank: int
    ple_layers: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_base: int
    ngram_divisor: int
    ngram_seed: int
    indexer_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress: int


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
    """method: bf16, fp16, nvfp4 (compressed-tensors) or modelopt (NVFP4 and FP8 weights per module, read from the
    tensors of each module)."""

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


def load_config(path: str | Path, yarn_factor: float | None = None) -> ModelConfig:
    """Read config.json (and generation_config.json, if present) from a checkpoint folder. yarn_factor: extend the
    context with YaRN by this factor over the checkpoint's original length (see parse_config)."""
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    generation = None
    if (path / "generation_config.json").is_file():
        generation = json.loads((path / "generation_config.json").read_text())
    return parse_config(raw, path=path, generation=generation, yarn_factor=yarn_factor)


def yarn_attention_factor(factor: float, mscale: float | None = None, mscale_all_dim: float | None = None) -> float:
    """YaRN's attention factor for cos and sin when the config does not give one (transformers' get_mscale)."""

    def get(scale: float, m: float = 1.0) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    if mscale and mscale_all_dim:
        return get(factor, mscale) / get(factor, mscale_all_dim)
    return get(factor)


def with_yarn(text_raw: dict, factor: float) -> dict:
    """A copy of a text config with YaRN at factor over its original length (the length of an existing YaRN config,
    else max_position_embeddings), as the model cards describe for longer contexts."""
    out = dict(text_raw)
    key = "rope_parameters" if "rope_parameters" in out or "rope_scaling" not in out else "rope_scaling"
    rope = dict(out.get(key) or {})
    if rope.get("rope_type", rope.get("type", "default")) == "yarn":
        original = int(rope.get("original_max_position_embeddings") or out.get("max_position_embeddings") or 0)
    else:
        original = int(out.get("max_position_embeddings") or 0)
    for k in ("type", "attention_factor", "mscale", "mscale_all_dim"):
        rope.pop(k, None)
    rope.update(rope_type="yarn", factor=float(factor), original_max_position_embeddings=original)
    if "rope_theta" not in rope and "rope_theta" in out:
        rope["rope_theta"] = out["rope_theta"]
    out[key] = rope
    return out


def parse_config(raw: dict, path: Path | None = None, generation: dict | None = None,
                 yarn_factor: float | None = None) -> ModelConfig:
    reasons: list[str] = []
    architectures = raw.get("architectures") or []
    architecture = architectures[0] if architectures else ""
    if architecture not in ARCHITECTURES:
        reasons.append(f"architecture {architecture or '(none)'} is not supported; supported: {', '.join(ARCHITECTURES)}")
    text_raw = raw.get("text_config", raw)
    if yarn_factor:
        if yarn_factor < 1:
            reasons.append(f"the YaRN factor {yarn_factor} is below 1")
        text_raw = with_yarn(text_raw, yarn_factor)
    text = _parse_text(text_raw, generation, reasons)
    vision = None
    if architecture in MULTIMODAL_ARCHITECTURES and not raw.get("language_model_only", False):
        vision = _parse_vision(raw.get("vision_config"), reasons)
    dtype = raw.get("dtype") or raw.get("torch_dtype") or text_raw.get("dtype") or text_raw.get("torch_dtype")
    quant = _parse_quant(raw.get("quantization_config"), dtype, reasons)
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
    if not raw.get("attn_output_gate", raw.get("output_gate_type") == "sigmoid"):
        reasons.append("attention without an output gate is not supported")
    if raw.get("attention_bias", False):
        reasons.append("attention bias is not supported")

    head_dim = _require(raw, "head_dim", reasons)
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    yarn = {}
    if rope_type == "yarn":
        yarn_f = float(rope.get("factor") or 0)
        original = int(rope.get("original_max_position_embeddings") or raw.get("max_position_embeddings") or 0)
        if yarn_f < 1 or original <= 0:
            reasons.append("the YaRN rope parameters need a factor of at least 1 and the original length")
        attention = rope.get("attention_factor")
        if attention is None:
            attention = yarn_attention_factor(yarn_f, rope.get("mscale"), rope.get("mscale_all_dim"))
        yarn = dict(rope_type="yarn", rope_factor=yarn_f, rope_original_max=original,
                    rope_attention_factor=float(attention), rope_beta_fast=float(rope.get("beta_fast") or 32),
                    rope_beta_slow=float(rope.get("beta_slow") or 1), rope_truncate=bool(rope.get("truncate", True)))
    elif rope_type != "default":
        reasons.append(f"rope type {rope_type!r} is not supported (only default and yarn)")
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
    gdn_k_dim = _require(raw, "linear_key_head_dim", reasons)
    gdn_v_dim = _require(raw, "linear_value_head_dim", reasons)
    conv_kernel = _require(raw, "linear_conv_kernel_dim", reasons)
    if not 2 <= conv_kernel <= 4:
        reasons.append(f"linear_conv_kernel_dim {conv_kernel} is not supported (2 to 4)")
    ssm_dtype = raw.get("mamba_ssm_dtype", "float32")
    if ssm_dtype != "float32":
        reasons.append(f"mamba_ssm_dtype {ssm_dtype!r} is not supported (only float32)")
    if raw.get("mtp_num_hidden_layers", 0) and raw.get("mtp_use_dedicated_embeddings", False):
        reasons.append("MTP with dedicated embeddings is not supported")

    num_experts = int(raw.get("num_experts") or 0)
    moe = {}
    if num_experts:
        moe = dict(num_experts=num_experts, experts_per_token=_require(raw, "num_experts_per_tok", reasons),
                   moe_intermediate_size=_require(raw, "moe_intermediate_size", reasons),
                   shared_expert_intermediate_size=_require(raw, "shared_expert_intermediate_size", reasons))
        if not 0 < moe["experts_per_token"] <= num_experts:
            reasons.append(f"num_experts_per_tok {moe['experts_per_token']} is not between 1 and {num_experts}")
        if raw.get("norm_topk_prob", True) is False and model_type not in ("qwen4_exp_text", "qwen4_exp"):
            reasons.append("MoE routing without normalized top-k weights is not supported")
        sparse_step = raw.get("decoder_sparse_step", 1)
        if sparse_step not in (None, 1) or raw.get("mlp_only_layers"):
            reasons.append("MoE models with dense layers between the MoE layers are not supported")

    exp = None
    if model_type in ("qwen4_exp_text", "qwen4_exp"):
        exp = _parse_exp(raw, reasons)
        if num_experts:
            moe["norm_topk_prob"] = bool(raw.get("norm_topk_prob", True))

    eos = _token_ids(raw.get("eos_token_id"))
    if generation:
        eos = tuple(dict.fromkeys(eos + _token_ids(generation.get("eos_token_id"))))

    return TextConfig(
        hidden_size=_require(raw, "hidden_size", reasons),
        intermediate_size=int(raw.get("intermediate_size") or 0) if num_experts else _require(raw, "intermediate_size",
                                                                                             reasons),
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
        gdn_k_dim=gdn_k_dim,
        gdn_v_dim=gdn_v_dim,
        conv_kernel=conv_kernel,
        max_position_embeddings=_require(raw, "max_position_embeddings", reasons),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        eos_token_ids=eos,
        exp=exp,
        **moe,
        **yarn,
    )


def _parse_exp(raw: dict, reasons: list[str]) -> ExpConfig:
    ple_ids = sorted(set(raw.get("ple_layer_ids") or ()))
    num_layers = int(raw.get("num_hidden_layers") or 0)
    if raw.get("indexer_kv_heads", 1) != 1:
        reasons.append("the QSA indexer needs one key head")
    if raw.get("output_gate_type", "sigmoid") != "sigmoid":
        reasons.append(f"attention output gate {raw.get('output_gate_type')!r} is not supported (only sigmoid)")
    return ExpConfig(
        hc_count=int(raw.get("hc_count", 4)),
        hc_lowrank=int(raw.get("hc_lowrank", 320)),
        ple_layers=tuple(i for i in range(num_layers) if i + 1 in ple_ids),
        ple_embed_dim=int(raw.get("ple_embed_dim") or raw.get("hidden_size") or 0),
        ple_conv_kernel=int(raw.get("ple_conv_kernel_size", 4)),
        ngram_size=int(raw.get("ngram_size", 3)),
        heads_per_ngram=int(raw.get("heads_per_ngram", 8)),
        ngram_vocab_base=int(raw.get("ngram_vocab_size_base", 20_000_000)),
        ngram_divisor=int(raw.get("make_ngram_vocab_size_divisible_by", 128)),
        ngram_seed=int(raw.get("seed", 1234)),
        indexer_heads=int(raw.get("indexer_n_heads") or 0),
        indexer_kv_heads=int(raw.get("indexer_kv_heads") or 0),
        indexer_head_dim=int(raw.get("indexer_head_dim") or 0),
        indexer_budget=int(raw.get("indexer_budget") or 0),
        indexer_compress=int(raw.get("indexer_compress_ratio") or 0),
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
    if method == "modelopt":
        return _parse_modelopt(raw, reasons)
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


def _parse_modelopt(raw: dict, reasons: list[str]) -> QuantConfig:
    """A ModelOpt checkpoint (quant_method modelopt): NVFP4 or mixed precision. The loader finds the format of each
    module from its tensors; this checks that every quantized layer uses a format seventy reads."""
    algo = raw.get("quant_algo")
    layers = raw.get("quantized_layers") or {}
    algos = {str(v.get("quant_algo")) for v in layers.values() if isinstance(v, dict)} if layers else {str(algo)}
    unknown = sorted(a for a in algos if a not in MODELOPT_ALGOS)
    if unknown:
        reasons.append(f"ModelOpt formats {', '.join(unknown)} are not supported (supported: {', '.join(MODELOPT_ALGOS)})")
    for name, group in (raw.get("config_groups") or {}).items():
        size = (group.get("weights") or {}).get("group_size")
        if size not in (None, 16):
            reasons.append(f"quantization group {name} needs group size 16")
    return QuantConfig(method="modelopt", group_size=16, ignore=tuple(raw.get("ignore") or ()))


def kernel_problems(cfg: ModelConfig) -> list[str]:
    """Why the CUDA kernels cannot run the model (empty when they can): the attention kernels are built for heads of
    ATTENTION_HEAD_DIM, the Gated DeltaNet kernels for heads of GDN_HEAD_DIM. parse_config accepts other sizes, which
    the PyTorch path (tests on the CPU) runs."""
    t, out = cfg.text, []
    if t.head_dim != ATTENTION_HEAD_DIM:
        out.append(f"attention head size {t.head_dim} is not supported (the attention kernels need {ATTENTION_HEAD_DIM})")
    if (t.gdn_k_dim, t.gdn_v_dim) != (GDN_HEAD_DIM, GDN_HEAD_DIM):
        out.append(f"Gated DeltaNet head sizes {t.gdn_k_dim}/{t.gdn_v_dim} are not supported (the kernels need "
                   f"{GDN_HEAD_DIM})")
    return out


def has_mtp(path: str | Path) -> bool:
    """Whether a checkpoint folder holds an MTP layer (mtp.fc, or for Qwen4-Exp mtp.fc_hidden, and mtp.layers.0),
    from the tensor names of its index file, else of the header of its model.safetensors."""
    p = Path(path)
    try:
        if (p / "model.safetensors.index.json").is_file():
            names = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
        else:
            with open(p / "model.safetensors", "rb") as f:
                names = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    mlp = ("mtp.layers.0.mlp.down_proj.weight", "mtp.layers.0.mlp.experts.down_proj", "mtp.layers.0.mlp.experts.0.down_proj.weight",
           "mtp.layers.0.mlp.experts.0.down_proj.weight_packed")
    fc = "mtp.fc.weight" in names or "mtp.fc_hidden.weight" in names
    return fc and any(m in names for m in mlp)
