"""Checkpoints on this machine and downloads from the Hugging Face Hub. No PyTorch.

A folder is a target model when seventy.config accepts its config.json, and a drafter when its architecture is
DFlash2DraftModel. A target's details tell whether it holds an MTP layer (the drafter option "mtp"). Folders come from
the Hugging Face cache (models--*/snapshots/*) and from folders the user adds.
Downloads use huggingface_hub with the endpoint from the panel settings or HF_ENDPOINT (default: the public Hub).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import UnsupportedModel, has_mtp, load_config
from ..memory import checkpoint_bytes


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return home / "hub"


@dataclass
class ModelInfo:
    path: str
    name: str
    kind: str
    ok: bool
    size: int
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def check(path: str | Path) -> ModelInfo:
    """What a folder holds and whether seventy can serve it (kind: target, drafter or unknown)."""
    p = Path(path)
    name = _name(p)
    cfg_file = p / "config.json"
    if not cfg_file.is_file():
        return ModelInfo(str(p), name, "unknown", False, 0, ["the folder has no config.json"])
    try:
        raw = json.loads(cfg_file.read_text())
    except (OSError, ValueError) as e:
        return ModelInfo(str(p), name, "unknown", False, 0, [f"config.json cannot be read: {e}"])
    size = checkpoint_bytes(p)
    if "DFlash2DraftModel" in (raw.get("architectures") or []):
        d = raw.get("dflash_config") or {}
        return ModelInfo(str(p), name, "drafter", size > 0, size, [] if size else ["no safetensors files"],
                         {"block_size": d.get("block_size"), "target_layers": d.get("target_layer_ids"),
                          "layers": raw.get("num_hidden_layers")})
    try:
        cfg = load_config(p)
    except UnsupportedModel as e:
        return ModelInfo(str(p), name, "target", False, size, list(e.reasons))
    reasons = []
    if not size:
        reasons.append("no safetensors files")
    if not (p / "tokenizer.json").is_file():
        reasons.append("no tokenizer.json")
    t = cfg.text
    return ModelInfo(str(p), name, "target", not reasons, size, reasons, {
        "architecture": cfg.architecture, "layers": t.num_layers, "hidden": t.hidden_size,
        "weights": cfg.quant.method, "vision": cfg.vision is not None,
        "max_context": t.max_position_embeddings, "kv_heads": t.num_kv_heads, "mtp": has_mtp(p),
    })


def _name(p: Path) -> str:
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
    return p.name


def scan(extra_dirs: list[str] = ()) -> list[ModelInfo]:
    """Every checkpoint folder in the Hugging Face cache and in extra_dirs (and their subfolders, one level)."""
    folders: list[Path] = []
    cache = hub_cache()
    if cache.is_dir():
        for repo in sorted(cache.glob("models--*")):
            snaps = sorted((repo / "snapshots").glob("*"), key=lambda s: s.stat().st_mtime, reverse=True)
            folders += [s for s in snaps[:1] if (s / "config.json").is_file()]
    for d in extra_dirs:
        base = Path(d).expanduser()
        if (base / "config.json").is_file():
            folders.append(base)
        elif base.is_dir():
            folders += [c for c in sorted(base.iterdir()) if (c / "config.json").is_file()]
    seen, out = set(), []
    for f in folders:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(check(f))
    return out


@dataclass
class Download:
    id: str
    repo: str
    revision: str | None
    endpoint: str | None
    state: str = "queued"
    total: int = 0
    done: int = 0
    path: str | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class Downloads:
    """Hub downloads in background threads, one at a time per repository."""

    def __init__(self):
        self.items: dict[str, Download] = {}
        self.lock = threading.Lock()

    def start(self, repo: str, revision: str | None = None, endpoint: str | None = None,
              token: str | None = None) -> Download:
        repo = repo.strip()
        if repo.count("/") != 1 or any(c.isspace() for c in repo):
            raise ValueError("a repository id looks like owner/name")
        with self.lock:
            for d in self.items.values():
                if d.repo == repo and d.state in ("queued", "running"):
                    return d
            d = Download(uuid.uuid4().hex[:10], repo, revision, endpoint)
            self.items[d.id] = d
        threading.Thread(target=self._run, args=(d, token), daemon=True, name=f"download-{repo}").start()
        return d

    def _run(self, d: Download, token: str | None) -> None:
        from huggingface_hub import HfApi, snapshot_download

        d.state = "running"
        try:
            api = HfApi(endpoint=d.endpoint or None, token=token or None)
            info = api.model_info(d.repo, revision=d.revision, files_metadata=True)
            d.total = sum(s.size or 0 for s in info.siblings or [])
            stop = threading.Event()
            watcher = threading.Thread(target=self._watch, args=(d, stop), daemon=True)
            watcher.start()
            try:
                d.path = snapshot_download(d.repo, revision=d.revision, endpoint=d.endpoint or None,
                                           token=token or None)
            finally:
                stop.set()
            d.done = d.total
            d.state = "done"
        except Exception as e:
            d.state, d.error = "failed", f"{type(e).__name__}: {e}"

    @staticmethod
    def _watch(d: Download, stop: threading.Event) -> None:
        repo_dir = hub_cache() / ("models--" + d.repo.replace("/", "--"))
        while not stop.wait(1.0):
            total = 0
            for f in (repo_dir / "blobs").glob("*") if (repo_dir / "blobs").is_dir() else ():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
            d.done = min(total, d.total) if d.total else total

    def list(self) -> list[dict]:
        with self.lock:
            return [d.to_dict() for d in sorted(self.items.values(), key=lambda x: -x.started)]
