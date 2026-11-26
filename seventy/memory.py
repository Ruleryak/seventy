"""The memory model shared by the engine and the control panel (no PyTorch import).

Per GPU, the engine gives the KV cache and the state slots what is left of utilization x total after loading
(weights, the drafter, and other memory in use: CUDA context, NCCL and all-reduce buffers) and after the activation
reserve, which also holds the CUDA graph pools. The panel predicts the same numbers before a start, from the
checkpoint sizes and from what earlier starts measured.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig

from .split import MLP_ALIGN, VOCAB_ALIGN, balanced, kv_groups, max_head_split

KV_DTYPES = ("int8", "fp16")
UNEVEN_MAX_GPUS = 4
"""Uneven splits (split.balanced, split.HeadSplit) are tested on up to 4 GPUs; with more GPUs, every count must
split evenly."""


def kv_split(cfg: ModelConfig, tp: int, block_size: int) -> int:
    """How many ranks share each KV head and keep one share of its tokens each (1: every rank keeps all tokens of
    its KV heads). Qwen4-Exp models on more GPUs than KV heads split when the ranks form equal groups per KV head
    (split.kv_groups) and the group size divides the indexer's block of tokens and the cache block, unless
    SEVENTY_KV_SPLIT=0. Rank i of a group keeps the tokens at cache slots s with s % group == i."""
    t = cfg.text
    if t.exp is None or tp < 2 or os.environ.get("SEVENTY_KV_SPLIT", "1") == "0":
        return 1
    groups = kv_groups(t.num_heads, t.num_kv_heads, tp)
    if groups is None:
        return 1
    g = len(groups[0])
    return g if t.exp.indexer_compress % g == 0 and block_size % g == 0 else 1


def kv_bytes_per_block(cfg: ModelConfig, block_size: int, kv_dtype: str, tp: int = 1, extra_layers: int = 0,
                       split: int = 1) -> int:
    """Bytes of one KV block on one rank: K and V of the local KV heads of every full-attention layer, and of
    extra_layers more (the MTP drafter's attention layer), as INT8 values plus one FP32 scale per token and head,
    or FP16, for one share of the block's tokens when split ranks share the KV heads (kv_split). Qwen4-Exp models
    add the token's rotary positions and the FP16 pooled indexer key of every attention layer per block of
    indexer_compress tokens (on every rank)."""
    t = cfg.text
    per_vec = t.head_dim + 4 if kv_dtype == "int8" else 2 * t.head_dim
    kv_heads = max_head_split(t.num_heads, t.num_kv_heads, tp)[0]
    kv = (len(t.attention_layers) + extra_layers) * 2 * (block_size // split) * kv_heads * per_vec
    if t.exp is not None:
        e = t.exp
        per_layer = e.indexer_head_dim * 2
        kv += block_size * 3 * 4 + (len(t.attention_layers) + extra_layers) * per_layer * (
            block_size // e.indexer_compress)
    return kv


def prefix_entries(max_num_seqs: int, prefix_cache: bool, cache_entries: int = 0) -> int:
    """State slots the prefix cache keeps besides the running streams (cache_entries 0: three per stream)."""
    return (cache_entries or 3 * max_num_seqs) if prefix_cache else 0


def state_bytes_per_slot(cfg: ModelConfig, tp: int = 1, extra_layers: int = 0) -> int:
    """Bytes of one Gated DeltaNet state slot on one rank: the FP32 recurrent state and the FP16 convolution
    state of every linear-attention layer (Qwen4-Exp: also the PLE tokens and convolution inputs and the ring of raw
    indexer keys) (of the fullest rank when the heads do not split evenly)."""
    t = cfg.text
    hk, hv = max_head_split(t.gdn_v_heads, t.gdn_k_heads, tp)
    conv = 2 * hk * t.gdn_k_dim + hv * t.gdn_v_dim
    state = len(t.linear_layers) * (hv * t.gdn_k_dim * t.gdn_v_dim * 4 + conv * (t.conv_kernel - 1) * 2)
    if t.exp is not None:
        from .engine.cache import IDX_RING

        e = t.exp
        state += (e.ngram_size - 1) * 8 + len(e.ple_layers) * (e.ple_conv_kernel - 1) * e.ngram_size * e.hc_count * t.hidden_size * 4
        state += (len(t.attention_layers) + extra_layers) * IDX_RING * e.indexer_head_dim * 2
    return state


@dataclass(frozen=True)
class DrafterShape:
    layers: int
    kv_heads: int
    head_dim: int
    ring: int
    heads: int = 0
    intermediate: int = 0

    @staticmethod
    def from_checkpoint(path: str | Path) -> "DrafterShape":
        raw = json.loads((Path(path) / "config.json").read_text())
        window = int(raw.get("sliding_window") or 2048)
        block = int(raw["dflash_config"]["block_size"])
        return DrafterShape(
            layers=int(raw["num_hidden_layers"]),
            kv_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]),
            ring=-(-(window + block) // 16) * 16,
            heads=int(raw["num_attention_heads"]),
            intermediate=int(raw.get("intermediate_size") or 0),
        )


def split_problems(cfg: ModelConfig, tp: int, drafter: DrafterShape | None = None) -> list[str]:
    """Why the model (and its drafter) cannot be split over tp GPUs (empty when it can). On up to UNEVEN_MAX_GPUS
    GPUs, counts that do not split evenly split into parts of whole units (split.balanced): each GPU needs at least
    one query head, and the MLP and the vocabulary need sizes that are multiples of their units."""
    t = cfg.text
    if tp > UNEVEN_MAX_GPUS:
        return _even_split_problems(cfg, tp, drafter)
    problems = []
    heads = [("attention heads", t.num_heads), ("Gated DeltaNet value heads", t.gdn_v_heads)]
    if cfg.vision is not None:
        heads.append(("vision heads", cfg.vision.num_heads))
    if drafter is not None and drafter.heads:
        heads.append(("drafter attention heads", drafter.heads))
    problems += [f"{total} {what} do not split over {tp} GPUs" for what, total in heads if total < tp]
    if t.is_moe:
        heads.append(("experts", t.num_experts))
    sizes = [("MLP columns", t.intermediate_size, MLP_ALIGN), ("vocabulary rows", t.vocab_size, VOCAB_ALIGN)]
    if t.is_moe:
        sizes.append(("shared expert columns", t.shared_expert_intermediate_size, MLP_ALIGN))
    if drafter is not None and drafter.intermediate:
        sizes.append(("drafter MLP columns", drafter.intermediate, MLP_ALIGN))
    for what, total, align in sizes:
        try:
            balanced(total, tp, 0, align)
        except ValueError:
            problems.append(f"{total} {what} do not split over {tp} GPUs (not a multiple of {align})")
    return problems


def _even_split_problems(cfg: ModelConfig, tp: int, drafter: DrafterShape | None) -> list[str]:
    """Why the model cannot be split evenly over tp GPUs: the rule for more than UNEVEN_MAX_GPUS GPUs."""
    t = cfg.text
    problems = [f"{total} {what} do not split over {tp} GPUs"
                for what, total in (("attention KV heads", t.num_kv_heads), ("attention heads", t.num_heads),
                                    ("Gated DeltaNet value heads", t.gdn_v_heads),
                                    ("Gated DeltaNet key heads", t.gdn_k_heads), ("vocabulary rows", t.vocab_size),
                                    ("MLP columns", t.intermediate_size))
                if total % tp]
    if cfg.vision is not None and (cfg.vision.num_heads % tp or cfg.vision.intermediate_size % tp):
        problems.append(f"the vision tower does not split over {tp} GPUs")
    if drafter is not None and drafter.kv_heads % tp:
        problems.append(f"the drafter's {drafter.kv_heads} KV heads do not split over {tp} GPUs")
    return problems


def drafter_bytes_per_slot(shape: DrafterShape, tp: int = 1) -> int:
    """Bytes of one slot's drafter context ring on one rank (FP16 K and V of the KV heads of the fullest rank)."""
    kv = max_head_split(shape.heads, shape.kv_heads, tp)[0] if shape.heads else -(-shape.kv_heads // tp)
    return 2 * shape.layers * shape.ring * kv * shape.head_dim * 2


def _shard_files(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [path / f for f in files]
    return sorted(path.glob("*.safetensors"))


def _gguf_tensors(path: Path) -> list[tuple[str, int, int]]:
    """(checkpoint-like name, values, bytes) of the tensors of a folder's GGUF files: routed experts as
    *.mlp.experts.*, the PLE table as *.ngram_embedding.shard_0."""
    from .gguf import GGUFFile

    out = []
    for f in sorted(path.glob("*.gguf")):
        for name, t in GGUFFile(f).tensors.items():
            like = (f"{name}.mlp.experts." if "_exps" in name
                    else "per_layer.ngram_embedding.shard_0" if name.startswith("per_layer_token_embd") else name)
            values = 1
            for d in t.dims:
                values *= d
            out.append((like, values, t.data.size))
    return out


def _tensor_bytes(path: str | Path, keep) -> int:
    """Bytes of the tensors whose names keep(name) accepts, from the headers of the checkpoint's files (GGUF files:
    _gguf_tensors)."""
    import struct

    if any(Path(path).glob("*.gguf")) and not _shard_files(Path(path)):
        return sum(size for name, _, size in _gguf_tensors(Path(path)) if keep(name))
    total = 0
    for f in _shard_files(Path(path)):
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
        except (OSError, ValueError):
            continue
        total += sum(m["data_offsets"][1] - m["data_offsets"][0] for name, m in header.items()
                     if name != "__metadata__" and keep(name))
    return total


def host_table_bytes(path: str | Path) -> int:
    """Bytes of the tensors that stay in host memory: the n-gram embedding tables of Qwen4-Exp checkpoints
    (*.ngram_embedding.shard_*), from the headers of their files."""
    return _tensor_bytes(path, lambda name: ".ngram_embedding.shard_" in name)


def expert_pool() -> int:
    """GPU slots per MoE layer and GPU for offloaded experts (the expert cache, seventy/model/moe.py), out of the GPU
    share of --expert-offload: SEVENTY_EXPERT_POOL, default 8."""
    return int(os.environ.get("SEVENTY_EXPERT_POOL", "8"))


def expert_host_share(cfg: ModelConfig, tp: int, offload: float) -> float:
    """The share of a MoE model's routed experts in host memory with --expert-offload offload: the offloaded ones
    and those that take turns in the GPU slots of the expert cache (expert_pool)."""
    t = cfg.text
    per_gpu = max(1, t.num_experts // tp) if t.is_moe else 1
    hosted = round(per_gpu * offload)
    if hosted == 0:
        return 0.0
    pool = max(0, min(expert_pool(), per_gpu - hosted - 1))
    return (hosted + pool) / per_gpu


def routed_expert_bytes(path: str | Path) -> int:
    """Bytes of the routed experts of a MoE checkpoint (*.mlp.experts.*, also the MTP layer's), from the headers of
    its files: the part that --expert-offload can keep in host memory."""
    return _tensor_bytes(path, lambda name: ".mlp.experts." in name)


def checkpoint_bytes(path: str | Path) -> int:
    """Total size of a checkpoint's safetensors files (the files of its index, which may be in a subfolder), less
    the tensors that stay in host memory (host_table_bytes). A GGUF folder: its routed experts as stored, every
    other tensor but the PLE table at FP16 (the engine dequantizes them)."""
    files = _shard_files(Path(path))
    if not files and any(Path(path).glob("*.gguf")):
        return sum(size if ".mlp.experts." in name else 0 if ".ngram_embedding." in name else 2 * values
                   for name, values, size in _gguf_tensors(Path(path)))
    total = sum(f.stat().st_size for f in files if f.is_file())
    return total - host_table_bytes(path) if total else total


def tied_head_bytes(cfg: ModelConfig, tp: int = 1) -> int:
    """Bytes on one rank of the output head that the checkpoint files do not hold. With tied embeddings
    (tie_word_embeddings), the files store the embedding once, and the engine keeps the head as its own FP16 copy of
    the rank's vocab rows: about 1.2 GiB for a 2560-wide model with a 248K vocabulary on one GPU."""
    t = cfg.text
    if not t.tie_word_embeddings:
        return 0
    v0, v1 = balanced(t.vocab_size, tp, 0, VOCAB_ALIGN)
    return (v1 - v0) * t.hidden_size * 2


def kv_shortage(free: int, state: int, streams: int, entries: int) -> str:
    """Why the engine cannot start: free is what one rank has after the weights, the drafter, the other memory in
    use and the activation reserve; state is what the state slots of the streams and the prefix-cache entries take
    from it. The message names the options that free memory."""
    gib = 2**30
    if free <= 0:
        why = (f"the weights and the activation reserve take {-free / gib:.1f} GiB more than the budget of the "
               f"smallest GPU (--gpu-memory-utilization)")
    else:
        why = (f"{free / gib:.1f} GiB is left after the weights and the activation reserve, and the state of "
               f"{streams} streams and {entries} prefix-cache entries takes {state / gib:.1f} GiB of it")
    return (f"not enough GPU memory for the KV cache: {why}. Use fewer streams (--max-num-seqs), no prefix cache "
            f"(--no-prefix-cache), more GPUs (--gpus), or for a mixture-of-experts model a part of the experts in "
            f"host memory (--expert-offload)")


@dataclass(frozen=True)
class Plan:
    """What one rank holds and how many KV tokens fit."""

    used: int
    reserve: int
    state: int
    kv_per_block: int
    blocks: int
    block_size: int

    @property
    def kv_tokens(self) -> int:
        return self.blocks * self.block_size


def plan(cfg: ModelConfig, gpu_bytes: int, used: int, tp: int, slots: int, block_size: int = 16,
         kv_dtype: str = "int8", utilization: float = 0.9, reserve: int = 3 << 30, drafter: DrafterShape | None = None,
         max_model_len: int | None = None, extra_layers: int = 0) -> Plan:
    """The KV blocks left on the tightest rank, as the engine computes them. gpu_bytes: the memory CUDA reports;
    used: what is in use after loading (weights, drafter, other); slots: state slots including prefix-cache
    entries (the scratch slot is added here); extra_layers: attention layers in the KV cache besides the model's
    (1 for the MTP drafter)."""
    per_slot = state_bytes_per_slot(cfg, tp) + (drafter_bytes_per_slot(drafter, tp) if drafter else 0)
    state = (slots + 1) * per_slot
    per_block = kv_bytes_per_block(cfg, block_size, kv_dtype, tp, extra_layers, kv_split(cfg, tp, block_size))
    free = int(gpu_bytes * utilization) - used - reserve - state
    blocks = max(0, free // per_block)
    if max_model_len:
        blocks = min(blocks, -(-max_model_len // block_size) * slots)
    return Plan(used, reserve, state, per_block, blocks, block_size)
