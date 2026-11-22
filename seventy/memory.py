"""The memory model shared by the engine and the control panel (no PyTorch import).

Per GPU, the engine gives the KV cache and the state slots what is left of utilization x total after loading
(weights, the drafter, and other memory in use: CUDA context, NCCL and all-reduce buffers) and after the activation
reserve, which also holds the CUDA graph pools. The panel predicts the same numbers before a start, from the
checkpoint sizes and from what earlier starts measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig

KV_DTYPES = ("int8", "fp16")


def kv_bytes_per_block(cfg: ModelConfig, block_size: int, kv_dtype: str, tp: int = 1) -> int:
    """Bytes of one KV block on one rank: K and V of the local KV heads of every full-attention layer
    (INT8 values plus one FP32 scale per token and head, or FP16)."""
    t = cfg.text
    per_vec = t.head_dim + 4 if kv_dtype == "int8" else 2 * t.head_dim
    return len(t.attention_layers) * 2 * block_size * (t.num_kv_heads // tp) * per_vec


def prefix_entries(max_num_seqs: int, prefix_cache: bool, cache_entries: int = 0) -> int:
    """State slots the prefix cache keeps besides the running streams (cache_entries 0: three per stream)."""
    return (cache_entries or 3 * max_num_seqs) if prefix_cache else 0


def state_bytes_per_slot(cfg: ModelConfig, tp: int = 1) -> int:
    """Bytes of one Gated DeltaNet state slot on one rank: the FP32 recurrent state and the FP16 convolution
    state of every linear-attention layer."""
    t = cfg.text
    hv = t.gdn_v_heads // tp
    conv = (2 * t.gdn_k_heads * t.gdn_k_dim + t.gdn_v_heads * t.gdn_v_dim) // tp
    return len(t.linear_layers) * (hv * t.gdn_k_dim * t.gdn_v_dim * 4 + conv * (t.conv_kernel - 1) * 2)


@dataclass(frozen=True)
class DrafterShape:
    layers: int
    kv_heads: int
    head_dim: int
    ring: int

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
        )


def split_problems(cfg: ModelConfig, tp: int, drafter: DrafterShape | None = None) -> list[str]:
    """Why the model (and its drafter) cannot be split over tp GPUs (empty when it can): tensor parallel splits
    each of these counts into equal parts."""
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
    """Bytes of one slot's drafter context ring on one rank (FP16 K and V of the rank's KV heads)."""
    return 2 * shape.layers * shape.ring * (shape.kv_heads // tp) * shape.head_dim * 2


def checkpoint_bytes(path: str | Path) -> int:
    """Total size of a checkpoint's safetensors files."""
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))


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
         max_model_len: int | None = None) -> Plan:
    """The KV blocks left on the tightest rank, as the engine computes them. gpu_bytes: the memory CUDA reports;
    used: what is in use after loading (weights, drafter, other); slots: state slots including prefix-cache
    entries (the scratch slot is added here)."""
    per_slot = state_bytes_per_slot(cfg, tp) + (drafter_bytes_per_slot(drafter, tp) if drafter else 0)
    state = (slots + 1) * per_slot
    per_block = kv_bytes_per_block(cfg, block_size, kv_dtype, tp)
    free = int(gpu_bytes * utilization) - used - reserve - state
    blocks = max(0, free // per_block)
    if max_model_len:
        blocks = min(blocks, -(-max_model_len // block_size) * slots)
    return Plan(used, reserve, state, per_block, blocks, block_size)
