"""Read tensors from a safetensors checkpoint: one file, or a sharded set with an index file."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .nvfp4 import Nvfp4Weight


def csf_decode(fixed: torch.Tensor, exceptions: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """NVFP4 block scales [rows, cols] (FP8 E4M3 bytes as uint8) from their CSF code, as in the ModelOpt checkpoints
    of nvidia (weight_scale.nvfp4_csf_fixed and weight_scale.nvfp4_csf_exceptions). The rows go in blocks of
    rows / fixed.shape[0]. Each row of fixed holds the base byte of each row of its block, then one 4-bit offset per
    scale (the first scale in the low nibble): scale = base + offset. Each exception is (scale << 24) | index into
    the flat scales, for scales that the base and offset do not reach. Lossless; checked byte for byte against the
    plain scales of the same tensors."""
    blocks = fixed.shape[0]
    per = rows // blocks
    if per * blocks != rows or fixed.shape[1] != per + per * cols // 2:
        raise ValueError(f"CSF code {tuple(fixed.shape)} does not match scales [{rows}, {cols}]")
    fixed = fixed.view(torch.uint8)
    base = fixed[:, :per].to(torch.int32)
    body = fixed[:, per:]
    offsets = torch.stack(((body & 15), (body >> 4)), dim=-1).reshape(blocks, per, cols).to(torch.int32)
    out = (base[:, :, None] + offsets).reshape(-1)
    exc = exceptions.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    out[exc & 0xFFFFFF] = (exc >> 24).to(torch.int32)
    return out.to(torch.uint8).view(rows, cols)


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

    def kind(self, module: str) -> str:
        """The weight format of a linear module: "nvfp4" (compressed-tensors: weight_packed, weight_scale and
        weight_global_scale), "modelopt-nvfp4" (weight as packed uint8, weight_scale, weight_scale_2; the block
        scales may be CSF-coded), "fp8" (FP8 E4M3 weight with one FP32 scale per tensor or per row), "mxfp8" (FP8
        E4M3 weight with one power-of-two scale per 32 columns) or "dense"."""
        if f"{module}.weight_packed" in self._files:
            return "nvfp4"
        if f"{module}.weight_scale_2" in self._files:
            return "modelopt-nvfp4"
        if f"{module}.weight_scale" in self._files:
            weight = self._slice(f"{module}.weight").get_dtype()
            scale = self._slice(f"{module}.weight_scale").get_dtype()
            if weight == "F8_E4M3":
                return "mxfp8" if scale == "U8" else "fp8"
        return "dense"

    def is_nvfp4(self, module: str) -> bool:
        return self.kind(module) in ("nvfp4", "modelopt-nvfp4")

    def out_rows(self, module: str) -> int:
        """Rows (output features) of a linear module's weight, in any format."""
        name = f"{module}.weight_packed" if f"{module}.weight_packed" in self._files else f"{module}.weight"
        return self.shape(name)[0]

    def _slice(self, name: str):
        return self._handle(name).get_slice(name)

    def nvfp4(self, module: str) -> Nvfp4Weight:
        """The NVFP4 weight of a linear module, for example "model.language_model.layers.0.mlp.up_proj"."""
        return self.nvfp4_part(module)

    def nvfp4_part(
        self, module: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None
    ) -> Nvfp4Weight:
        """Rows and columns [start, stop) of an NVFP4 weight, read without loading the rest of it (CSF-coded block
        scales are decoded whole first). ModelOpt's weight_scale_2 multiplies where compressed-tensors'
        weight_global_scale divides, so the global scale of a ModelOpt weight is 1 / weight_scale_2."""
        modelopt = self.kind(module) == "modelopt-nvfp4"
        packed_name = f"{module}.weight" if modelopt else f"{module}.weight_packed"
        n, half_k = self.shape(packed_name)
        r0, r1 = rows if rows is not None else (0, n)
        c0, c1 = cols if cols is not None else (0, 2 * half_k)
        if c0 % 16 or c1 % 16:
            raise ValueError(f"column slice [{c0}, {c1}) of {module} is not aligned to groups of 16")
        packed = self._slice(packed_name)[r0:r1, c0 // 2 : c1 // 2]
        if f"{module}.weight_scale" in self._files:
            scales = self._slice(f"{module}.weight_scale")[r0:r1, c0 // 16 : c1 // 16]
        else:
            scales = csf_decode(self.tensor(f"{module}.weight_scale.nvfp4_csf_fixed"),
                                self.tensor(f"{module}.weight_scale.nvfp4_csf_exceptions"), n, 2 * half_k // 16)
            scales = scales[r0:r1, c0 // 16 : c1 // 16]
        if modelopt:
            global_scale = 1.0 / float(self.tensor(f"{module}.weight_scale_2").float().reshape(-1)[0])
        else:
            global_scale = float(self.tensor(f"{module}.weight_global_scale").reshape(-1)[0])
        return Nvfp4Weight(packed=packed.contiguous(), scales=scales.contiguous().view(torch.uint8),
                           global_scale=global_scale)

    def fp8_part(self, module: str, rows: tuple[int, int] | None = None,
                 cols: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(E4M3 bytes uint8 [rows, cols], FP32 scale per row) of an "fp8" module, as stored."""
        weight = self.dense_part(f"{module}.weight", rows=rows, cols=cols).view(torch.uint8)
        scale = self.tensor(f"{module}.weight_scale").float().reshape(-1)
        if scale.numel() == 1:
            return weight, scale.expand(weight.shape[0]).clone()
        r0, r1 = rows if rows is not None else (0, scale.numel())
        return weight, scale[r0:r1].clone()

    def mxfp8_rows(self, module: str, rows: tuple[int, int] | None = None,
                   cols: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """An "mxfp8" module as FP8 E4M3 with one scale per row: (E4M3 bytes uint8, FP32 row scale = the row's
        largest block scale). Each block's values move by its exponent below the largest, which rounds the smallest
        values of the lower blocks to E4M3 subnormals: on Flash-Next's weights 0.01 to 0.02 % of the values change,
        and the weights move by 3e-7 to 6e-7 relative."""
        weight = self.dense_part(f"{module}.weight", rows=rows, cols=cols).view(torch.float8_e4m3fn).float()
        scale = self.tensor(f"{module}.weight_scale").view(torch.uint8)
        r0, r1 = rows if rows is not None else (0, scale.shape[0])
        c0, c1 = cols if cols is not None else (0, 32 * scale.shape[1])
        if c0 % 32 or c1 % 32:
            raise ValueError(f"column slice [{c0}, {c1}) of {module} is not aligned to MXFP8 blocks of 32")
        e = scale[r0:r1, c0 // 32 : c1 // 32].float() - 127.0
        top = e.amax(dim=1, keepdim=True)
        n, k = weight.shape
        shifted = (weight.view(n, -1, 32) * torch.exp2(e - top)[..., None]).view(n, k)
        return shifted.to(torch.float8_e4m3fn).view(torch.uint8), torch.exp2(top.reshape(-1))

    def linear_part(self, module: str, rows: tuple[int, int] | None = None,
                    cols: tuple[int, int] | None = None) -> torch.Tensor:
        """Rows and columns [start, stop) of a linear module's weight that is not NVFP4: dense as stored, or FP8 and
        MXFP8 dequantized to FP32 (exact: E4M3 values times their scales)."""
        kind = self.kind(module)
        weight = self.dense_part(f"{module}.weight", rows=rows, cols=cols)
        if kind == "dense":
            return weight
        values = weight.float()
        scale = self.tensor(f"{module}.weight_scale")
        if kind == "fp8":
            scale = scale.float()
            if scale.numel() == 1:
                return values * scale.reshape(())
            r0, r1 = rows if rows is not None else (0, scale.shape[0])
            return values * scale.reshape(-1)[r0:r1, None]
        r0, r1 = rows if rows is not None else (0, scale.shape[0])
        c0, c1 = cols if cols is not None else (0, 32 * scale.shape[1])
        if c0 % 32 or c1 % 32:
            raise ValueError(f"column slice [{c0}, {c1}) of {module} is not aligned to MXFP8 blocks of 32")
        exponent = scale.view(torch.uint8)[r0:r1, c0 // 32 : c1 // 32].float() - 127.0
        return (values.view(values.shape[0], -1, 32) * torch.exp2(exponent)[..., None]).view(values.shape)

    def dense_part(
        self, name: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """Rows and columns [start, stop) of a dense tensor of two or more dimensions."""
        shape = self.shape(name)
        r0, r1 = rows if rows is not None else (0, shape[0])
        c0, c1 = cols if cols is not None else (0, shape[1])
        return self._handle(name).get_slice(name)[r0:r1, c0:c1]

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
