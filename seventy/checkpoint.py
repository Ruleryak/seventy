"""Read tensors from a safetensors checkpoint: one file, or a sharded set with an index file."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .nvfp4 import Nvfp4Weight


class Checkpoint:
    """Lazy access to the tensors of a checkpoint folder. Only the requested tensors or slices are read."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        single = self.path / "model.safetensors"
        self._files: dict[str, Path] = {}
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._files = {name: self.path / file for name, file in weight_map.items()}
        elif single.is_file():
            with safe_open(single, framework="pt") as f:
                self._files = {name: single for name in f.keys()}
        else:
            raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {self.path}")
        self._handles: dict[Path, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._files

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self._files if n.startswith(prefix))

    @property
    def text_prefix(self) -> str:
        """Prefix of the text model's tensors: "model.language_model." in multimodal checkpoints, else "model."."""
        if any(n.startswith("model.language_model.") for n in self._files):
            return "model.language_model."
        return "model."

    def _handle(self, name: str):
        try:
            file = self._files[name]
        except KeyError:
            raise KeyError(f"tensor {name} is not in the checkpoint at {self.path}") from None
        handle = self._handles.get(file)
        if handle is None:
            handle = safe_open(file, framework="pt")
            self._handles[file] = handle
        return handle

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._handle(name).get_slice(name).get_shape())

    def tensor(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        """Rows [start, stop) of a tensor (dimension 0)."""
        return self._handle(name).get_slice(name)[start:stop]

    def cols(self, name: str, start: int, stop: int) -> torch.Tensor:
        """Columns [start, stop) of a tensor of two or more dimensions (dimension 1)."""
        return self._handle(name).get_slice(name)[:, start:stop]

    def is_nvfp4(self, module: str) -> bool:
        return f"{module}.weight_packed" in self._files

    def nvfp4(self, module: str) -> Nvfp4Weight:
        """The NVFP4 weight of a linear module, for example "model.language_model.layers.0.mlp.up_proj"."""
        return Nvfp4Weight(
            packed=self.tensor(f"{module}.weight_packed"),
            scales=self.tensor(f"{module}.weight_scale").view(torch.uint8),
            global_scale=float(self.tensor(f"{module}.weight_global_scale").reshape(-1)[0]),
        )

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
