"""What a start would give: memory per GPU and the KV cache size, before the engine starts. No PyTorch.

The memory CUDA reports per GPU, the weights, the drafter and the other memory in use after loading come from what
an earlier start with the same model, drafter, GPU count, GPU memory size and KV format measured (profiles.json in
the panel home, written when the engine is ready). Without one: the checkpoint sizes divided over the GPUs, and fixed
guesses for the CUDA context and communication buffers. The rest is seventy.memory, the model the engine itself uses.
GPUs with different memory sizes can run one engine: every GPU gets the KV cache of the smallest GPU, and the plan
shows that GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import memory
from ..config import UnsupportedModel, load_config
from .hardware import Inventory, gpu_set_problems, nvlink_full
from .supervisor import Settings

RESERVE = 3 << 30
CUDA_HIDDEN = 288 << 20
OVERHEAD_GUESS = {1: 640 << 20}
OVERHEAD_GUESS_TP = 1 << 30


def gpu_memory(inv: Inventory, ids: list[int]) -> int:
    """The memory size (NVML) of the smallest GPU in ids: the GPU that sets the KV cache size."""
    known = {g.id: g for g in inv.gpus}
    return min(known[i].memory_total for i in ids)


def profile_key(s: Settings, gpu_bytes: int) -> str:
    return json.dumps([str(Path(s.model).resolve()), str(Path(s.draft_model).resolve()) if s.draft_model else None,
                       len(s.gpus), s.kv_cache_dtype, gpu_bytes])


class Profiles:
    def __init__(self, home: Path):
        self.path = Path(home) / "profiles.json"

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def get(self, s: Settings, gpu_bytes: int) -> dict | None:
        """What a start of s measured on GPUs whose smallest has gpu_bytes of memory (NVML)."""
        return self.load().get(profile_key(s, gpu_bytes))

    def put(self, s: Settings, gpu_bytes: int, measured: dict) -> None:
        data = self.load()
        data[profile_key(s, gpu_bytes)] = measured
        self.path.write_text(json.dumps(data, indent=1))


def plan(s: Settings, inv: Inventory, profiles: Profiles | None = None) -> dict:
    errors, warnings = list(gpu_set_problems(inv, s.gpus)), []
    tp = max(1, len(s.gpus))
    out: dict = {"ok": False, "errors": errors, "warnings": warnings, "tp": tp}
    try:
        cfg = load_config(s.model)
    except (OSError, ValueError) as e:
        reasons = e.reasons if isinstance(e, UnsupportedModel) else [str(e)]
        errors.extend(reasons)
        return out
    t = cfg.text
    for what, total in (("attention KV heads", t.num_kv_heads), ("attention heads", t.num_heads),
                        ("Gated DeltaNet value heads", t.gdn_v_heads), ("Gated DeltaNet key heads", t.gdn_k_heads),
                        ("vocabulary rows", t.vocab_size), ("MLP columns", t.intermediate_size)):
        if total % tp:
            errors.append(f"{total} {what} do not split over {tp} GPUs")
    if cfg.vision is not None and (cfg.vision.num_heads % tp or cfg.vision.intermediate_size % tp):
        errors.append(f"the vision tower does not split over {tp} GPUs")
    drafter = None
    if s.draft_model:
        try:
            drafter = memory.DrafterShape.from_checkpoint(s.draft_model)
            if drafter.kv_heads % tp:
                errors.append(f"the drafter's {drafter.kv_heads} KV heads do not split over {tp} GPUs")
        except (OSError, ValueError, KeyError) as e:
            errors.append(f"the drafter cannot be read: {e}")
    if s.kv_cache_dtype not in memory.KV_DTYPES:
        errors.append(f"KV cache format must be one of {', '.join(memory.KV_DTYPES)}")
    max_len = s.max_model_len or t.max_position_embeddings
    if max_len > t.max_position_embeddings:
        errors.append(f"the model allows at most {t.max_position_embeddings} tokens of context")
    if errors:
        return out
    known = {g.id: g for g in inv.gpus}
    nvml_bytes = gpu_memory(inv, s.gpus)
    measured = profiles.get(s, nvml_bytes) if profiles else None
    if measured and "overhead" in measured:
        gpu_bytes = int(measured.get("gpu_total", nvml_bytes - CUDA_HIDDEN))
        weights = int(measured["weights"]) + int(measured.get("drafter", 0))
        overhead = int(measured["overhead"])
        graphs = int(measured.get("graphs", 0))
    else:
        measured = None
        gpu_bytes = nvml_bytes - CUDA_HIDDEN
        weights = memory.checkpoint_bytes(s.model) // tp
        if s.draft_model:
            weights += memory.checkpoint_bytes(s.draft_model) // tp
        overhead = OVERHEAD_GUESS.get(tp, OVERHEAD_GUESS_TP)
        graphs = None
    slots = s.max_num_seqs + memory.prefix_entries(s.max_num_seqs, s.prefix_cache)
    p = memory.plan(cfg, gpu_bytes, weights + overhead, tp, slots, 16, s.kv_cache_dtype, s.gpu_memory_utilization,
                    RESERVE, drafter, max_len)
    out.update({
        "ok": p.blocks * 16 >= min(max_len, 1024),
        "measured": bool(measured),
        "gpu_bytes": gpu_bytes,
        "budget": int(gpu_bytes * s.gpu_memory_utilization),
        "weights": weights,
        "overhead": overhead,
        "graphs": graphs,
        "reserve": RESERVE,
        "state": p.state,
        "kv": p.blocks * p.kv_per_block,
        "kv_tokens": p.kv_tokens,
        "max_model_len": max_len,
        "nvlink_full": nvlink_full(inv, s.gpus) if tp > 1 else None,
    })
    if not out["ok"]:
        errors.append("the GPUs have no room left for a KV cache with these settings")
    elif p.kv_tokens < max_len:
        warnings.append(f"the KV cache holds {p.kv_tokens:,} tokens, less than one request at the maximum context "
                        f"({max_len:,}); long requests will wait or be cut")
    elif p.kv_tokens < max_len * s.max_num_seqs:
        warnings.append(f"{s.max_num_seqs} streams at full context need {max_len * s.max_num_seqs:,} tokens; the KV cache "
                        f"holds {p.kv_tokens:,}, so long requests share it (the newest may wait)")
    sizes = sorted({known[i].memory_total for i in s.gpus})
    if len(sizes) > 1:
        listed = " and ".join(f"{b / 2**30:.0f} GiB" for b in sizes)
        warnings.append(f"the GPUs have different memory sizes ({listed}); every GPU gets the KV cache of the smallest "
                        f"one, so the larger GPUs keep memory unused (this plan shows the smallest GPU)")
    if tp > 1 and not out["nvlink_full"]:
        warnings.append("not every GPU pair has NVLink; tensor parallel will be slower")
    if inv.ram_available and inv.ram_available < 16 << 30:
        warnings.append(f"only {inv.ram_available / 2**30:.0f} GiB of RAM is available; loading may be slow")
    return out
