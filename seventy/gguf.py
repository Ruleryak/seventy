"""GGUF files (llama.cpp): the header, the tensors as memory-mapped blocks, and their dequantization on the CPU.

A GGUF file is a header (magic, version 3, metadata key-value pairs, tensor infos: name, dimensions with the
fastest first, type, offset) and the tensor data after it, aligned (general.alignment, default 32). A model may come
in split files (split.count): the first has the metadata, every file has its own tensor infos. A tensor of type T
is a sequence of T's blocks along its first dimension (ne0), row after row.

The block formats and their dequantization follow ggml (ggml/src/ggml-quants.c and ggml-common.h of llama.cpp, MIT
License, Copyright (c) 2023-2026 The ggml authors), vectorized over the blocks with NumPy: F32, F16, BF16, Q4_0,
Q8_0, Q2_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS. The lookup tables of the
i-quants are in seventy/gguf_tables.py.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gguf_tables as tables

TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 30: "BF16",
              42: "Q2_0"}
"""The GGML types this module reads, by type id."""

BLOCKS = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q8_0": (32, 34), "Q2_0": (64, 18),
          "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74),
          "IQ2_S": (256, 82), "IQ3_XXS": (256, 98), "IQ3_S": (256, 110), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136)}
"""(values, bytes) of one block of each type."""

_VALUE_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass(frozen=True)
class Tensor:
    """One tensor of a GGUF file: dims with the fastest first (ggml ne), the type name, and its raw blocks as a
    uint8 array [rows, bytes per row] (memory-mapped) where rows is the product of dims[1:]."""

    name: str
    dims: tuple[int, ...]
    type: str
    data: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        """The PyTorch shape (slowest dimension first)."""
        return tuple(reversed(self.dims))

    def dequantize(self) -> np.ndarray:
        """FP32 values in the PyTorch shape."""
        return dequantize(self.data, self.type).reshape(self.shape)


class _Header:
    def __init__(self, buf: memoryview):
        self.buf, self.pos = buf, 0

    def take(self, fmt: str):
        v = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self) -> str:
        n = self.take("Q")
        s = bytes(self.buf[self.pos : self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def value(self, t: int):
        if t in _VALUE_TYPES:
            return self.take(_VALUE_TYPES[t])
        if t == 8:
            return self.string()
        if t == 9:
            item, n = self.take("I"), self.take("Q")
            if item in _VALUE_TYPES and item != 7:
                fmt = _VALUE_TYPES[item]
                size = struct.calcsize("<" + fmt)
                arr = np.frombuffer(self.buf, dtype=np.dtype("<" + fmt), count=n, offset=self.pos)
                self.pos += n * size
                return arr
            return [self.value(item) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


class GGUFFile:
    """One GGUF file, memory-mapped: metadata (dict; numeric arrays as NumPy arrays) and tensors (dict by name)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            self.map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buf = memoryview(self.map)
        h = _Header(buf)
        magic = h.take("4s")
        if magic != b"GGUF":
            raise ValueError(f"{self.path} is not a GGUF file")
        version = h.take("I")
        if version != 3:
            raise ValueError(f"{self.path}: GGUF version {version} is not supported (3 is)")
        n_tensors, n_kv = h.take("Q"), h.take("Q")
        self.metadata: dict = {}
        for _ in range(n_kv):
            key = h.string()
            self.metadata[key] = h.value(h.take("I"))
        infos = []
        for _ in range(n_tensors):
            name = h.string()
            dims = tuple(h.take("Q") for _ in range(h.take("I")))
            infos.append((name, dims, h.take("I"), h.take("Q")))
        align = int(self.metadata.get("general.alignment", 32))
        start = -(-h.pos // align) * align
        self.tensors: dict[str, Tensor] = {}
        for name, dims, t, offset in infos:
            kind = TYPE_NAMES.get(t)
            if kind is None:
                raise ValueError(f"{self.path}: tensor {name} has GGML type {t}, which seventy does not read")
            values, size = BLOCKS[kind]
            if dims[0] % values:
                raise ValueError(f"{self.path}: tensor {name} rows of {dims[0]} are not whole {kind} blocks")
            rows = int(np.prod(dims[1:], dtype=np.int64)) if len(dims) > 1 else 1
            per_row = dims[0] // values * size
            if start + offset + rows * per_row > len(self.map):
                raise ValueError(f"{self.path}: tensor {name} runs past the end of the file")
            data = np.frombuffer(self.map, dtype=np.uint8, count=rows * per_row, offset=start + offset)
            self.tensors[name] = Tensor(name, dims, kind, data.reshape(rows, per_row))


def open_model(path: str | Path) -> tuple[dict, dict[str, Tensor]]:
    """(metadata of the first file, tensors of all files) of a GGUF model: path is its first file (a split model's
    other files, -0000k-of-0000n.gguf, are found next to it)."""
    first = GGUFFile(path)
    tensors = dict(first.tensors)
    count = int(first.metadata.get("split.count", 1))
    if count > 1:
        p = Path(path)
        stem = p.name
        marker = f"-00001-of-{count:05d}.gguf"
        if not stem.endswith(marker):
            raise ValueError(f"{p} is a split GGUF model; give its first file (*{marker})")
        for k in range(2, count + 1):
            other = GGUFFile(p.with_name(stem[: -len(marker)] + f"-{k:05d}-of-{count:05d}.gguf"))
            tensors.update(other.tensors)
    return first.metadata, tensors


def _f16(b: np.ndarray) -> np.ndarray:
    return b.copy().view(np.float16).astype(np.float32)[..., 0]


def _u32(b: np.ndarray) -> np.ndarray:
    return b.copy().view("<u4")


def _grid_bytes(grid, n: int) -> np.ndarray:
    return np.array(grid, dtype="<u8" if n == 8 else "<u4").view(np.uint8).reshape(len(grid), n)


_IQ2XXS = _grid_bytes(tables.IQ2XXS_GRID, 8).astype(np.float32)
_IQ2XS = _grid_bytes(tables.IQ2XS_GRID, 8).astype(np.float32)
_IQ2S = _grid_bytes(tables.IQ2S_GRID, 8).astype(np.float32)
_IQ3XXS = _grid_bytes(tables.IQ3XXS_GRID, 4).astype(np.float32)
_IQ3S = _grid_bytes(tables.IQ3S_GRID, 4).astype(np.float32)
_KSIGNS = np.array(tables.KSIGNS_IQ2XS, dtype=np.uint8)
_KVALUES = np.array(tables.KVALUES_IQ4NL, dtype=np.float32)
_BITS8 = np.array(tables.KMASK_IQ2XS, dtype=np.uint8)


def _signs(bits: np.ndarray) -> np.ndarray:
    """+1 / -1 [..., 8] of the 8 bits of each byte of bits [...] (bit j: value j)."""
    return np.where((bits[..., None] & _BITS8) != 0, -1.0, 1.0).astype(np.float32)


def _scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 8 6-bit scales and mins [nb, 8] of the K-quant scales [nb, 12] (ggml get_scale_min_k4)."""
    q = scales.astype(np.int32)
    sc = np.empty(q.shape[:-1] + (8,), np.float32)
    mn = np.empty_like(sc)
    sc[..., :4] = q[..., 0:4] & 63
    mn[..., :4] = q[..., 4:8] & 63
    sc[..., 4:] = (q[..., 8:12] & 0xF) | ((q[..., 0:4] >> 6) << 4)
    mn[..., 4:] = (q[..., 8:12] >> 4) | ((q[..., 4:8] >> 6) << 4)
    return sc, mn


def _blocks(kind: str, b: np.ndarray) -> np.ndarray:
    """FP32 values [nb, block values] of raw blocks b [nb, block bytes]."""
    if kind == "Q8_0":
        return _f16(b[:, 0:2])[:, None] * b[:, 2:].view(np.int8).astype(np.float32)
    if kind == "Q4_0":
        q = b[:, 2:]
        v = np.concatenate((q & 0xF, q >> 4), axis=1).astype(np.float32) - 8
        return _f16(b[:, 0:2])[:, None] * v
    if kind == "Q2_0":
        q = b[:, 2:]
        codes = (q[:, :, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
        return _f16(b[:, 0:2])[:, None] * (codes.reshape(len(b), 64).astype(np.float32) - 1)
    if kind == "IQ4_NL":
        q = b[:, 2:]
        return _f16(b[:, 0:2])[:, None] * _KVALUES[np.concatenate((q & 0xF, q >> 4), axis=1)]
    if kind == "IQ4_XS":
        d = _f16(b[:, 0:2])
        sh = b[:, 2:4].copy().view("<u2")[:, 0].astype(np.int32)
        sl = b[:, 4:8].astype(np.int32)
        ib = np.arange(8)
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
        q = b[:, 8:].reshape(len(b), 8, 16)
        v = _KVALUES[np.concatenate((q & 0xF, q >> 4), axis=2)]
        return (d[:, None, None] * (ls - 32)[:, :, None] * v).reshape(len(b), 256)
    if kind == "Q4_K":
        d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        sc, mn = _scale_min_k4(b[:, 4:16])
        q = b[:, 16:].reshape(len(b), 4, 32)
        v = np.stack((q & 0xF, q >> 4), axis=2).reshape(len(b), 8, 32).astype(np.float32)
        return (d[:, None, None] * sc[:, :, None] * v - dmin[:, None, None] * mn[:, :, None]).reshape(len(b), 256)
    if kind == "Q5_K":
        d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        sc, mn = _scale_min_k4(b[:, 4:16])
        qh = b[:, 16:48]
        q = b[:, 48:].reshape(len(b), 4, 32)
        j = np.arange(4)[None, :, None]
        hi_lo = ((qh[:, None, :] >> (2 * j)) & 1) * 16
        hi_hi = ((qh[:, None, :] >> (2 * j + 1)) & 1) * 16
        v = np.stack(((q & 0xF) + hi_lo, (q >> 4) + hi_hi), axis=2).reshape(len(b), 8, 32).astype(np.float32)
        return (d[:, None, None] * sc[:, :, None] * v - dmin[:, None, None] * mn[:, :, None]).reshape(len(b), 256)
    if kind == "Q6_K":
        ql = b[:, 0:128].reshape(len(b), 2, 64)
        qh = b[:, 128:192].reshape(len(b), 2, 32)
        sc = b[:, 192:208].view(np.int8).astype(np.float32).reshape(len(b), 2, 8)
        d = _f16(b[:, 208:210])
        q1 = (ql[:, :, :32] & 0xF) | ((qh & 3) << 4)
        q2 = (ql[:, :, 32:] & 0xF) | (((qh >> 2) & 3) << 4)
        q3 = (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)
        q4 = (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)
        v = np.stack((q1, q2, q3, q4), axis=2).astype(np.float32) - 32
        s = sc.reshape(len(b), 2, 4, 2)
        scale = np.repeat(s, 16, axis=3)
        return (d[:, None, None, None] * scale * v).reshape(len(b), 256)
    if kind == "IQ2_XXS":
        d = _f16(b[:, 0:2])
        w = _u32(b[:, 2:66]).reshape(len(b), 8, 2)
        idx = b[:, 2:66].reshape(len(b), 8, 8)[:, :, :4]
        db = d[:, None] * (0.5 + (w[:, :, 1] >> 28).astype(np.float32)) * 0.25
        sgn = _KSIGNS[(w[:, :, 1, None] >> (7 * np.arange(4, dtype=np.uint32))) & 127]
        v = _IQ2XXS[idx] * _signs(sgn)
        return (db[:, :, None, None] * v).reshape(len(b), 256)
    if kind == "IQ2_XS":
        d = _f16(b[:, 0:2])
        q = b[:, 2:66].copy().view("<u2").reshape(len(b), 8, 4).astype(np.int32)
        s = b[:, 66:74].astype(np.float32)
        db = d[:, None, None] * (0.5 + np.stack((s.astype(np.int32) & 0xF, s.astype(np.int32) >> 4), axis=2)) * 0.25
        v = _IQ2XS[q & 511] * _signs(_KSIGNS[q >> 9])
        return (np.repeat(db, 2, axis=2)[:, :, :, None] * v).reshape(len(b), 256)
    if kind == "IQ2_S":
        d = _f16(b[:, 0:2])
        qs = b[:, 2:34].reshape(len(b), 8, 4).astype(np.int32)
        sg = b[:, 34:66].reshape(len(b), 8, 4)
        qh = b[:, 66:74].astype(np.int32)
        s = b[:, 74:82].astype(np.int32)
        idx = qs | (((qh[:, :, None] >> (2 * np.arange(4))) & 3) << 8)
        db = d[:, None, None] * (0.5 + np.stack((s & 0xF, s >> 4), axis=2)) * 0.25
        v = _IQ2S[idx] * _signs(sg)
        return (np.repeat(db, 2, axis=2)[:, :, :, None] * v).reshape(len(b), 256)
    if kind == "IQ3_XXS":
        d = _f16(b[:, 0:2])
        idx = b[:, 2:66].reshape(len(b), 8, 8)
        w = _u32(b[:, 66:98]).reshape(len(b), 8)
        db = d[:, None] * (0.5 + (w >> 28).astype(np.float32)) * 0.5
        sgn = _signs(_KSIGNS[(w[:, :, None] >> (7 * np.arange(4, dtype=np.uint32))) & 127])
        v = _IQ3XXS[idx].reshape(len(b), 8, 4, 8) * sgn
        return (db[:, :, None, None] * v).reshape(len(b), 256)
    if kind == "IQ3_S":
        d = _f16(b[:, 0:2])
        qs = b[:, 2:66].reshape(len(b), 8, 8).astype(np.int32)
        qh = b[:, 66:74].astype(np.int32)
        sg = b[:, 74:106].reshape(len(b), 8, 4)
        s = b[:, 106:110].astype(np.int32)
        idx = qs | (((qh[:, :, None] >> np.arange(8)) & 1) << 8)
        sc = np.stack((s & 0xF, s >> 4), axis=2).reshape(len(b), 8)
        db = d[:, None] * (1 + 2 * sc).astype(np.float32)
        v = _IQ3S[idx].reshape(len(b), 8, 4, 8) * _signs(sg)
        return (db[:, :, None, None] * v).reshape(len(b), 256)
    raise ValueError(f"no dequantization for {kind}")


def dequantize(data: np.ndarray, kind: str) -> np.ndarray:
    """FP32 values [rows, row values] of raw tensor rows data [rows, row bytes] of type kind."""
    rows = data.shape[0]
    if kind == "F32":
        return data.view("<f4").astype(np.float32).reshape(rows, -1)
    if kind == "F16":
        return data.view("<f2").astype(np.float32).reshape(rows, -1)
    if kind == "BF16":
        u = data.view("<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(rows, -1)
    values, size = BLOCKS[kind]
    blocks = np.ascontiguousarray(data).reshape(-1, size)
    return _blocks(kind, blocks).reshape(rows, -1)


_HC = {"norm": "hc_norm", "down": "input_mix_weight_down", "up": "input_mix_weight_up", "inject": "block_inject_weight"}
_LAYER = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",
    **{f"hc_attn_{k}.weight": f"attn_hyper_connection.{v}.weight" for k, v in _HC.items()},
    **{f"hc_ffn_{k}.weight": f"mlp_hyper_connection.{v}.weight" for k, v in _HC.items()},
}
_EXPERTS = {"ffn_gate_exps.weight": "gate", "ffn_up_exps.weight": "up", "ffn_down_exps.weight": "down"}


class GGUFCheckpoint:
    """A llama.cpp qwen4exp GGUF model (Qwen3.8-Flash-Next) read like a Hugging Face checkpoint (Checkpoint): its
    tensors under the checkpoint's names, dequantized to FP32 and back in the checkpoint's layout. The converter's
    changes that this undoes (as 1Cat-vLLM's GGUF adapter for the format does): norm weights stored as 1 + w (all
    RMS norms but the Gated DeltaNet output norm, the PLE norms included), A_log stored as -exp(A_log), the Gated DeltaNet value heads stored
    tile-major (value head r of key head g at r * key_heads + g instead of g * per_key + r: in_proj_qkv's value rows,
    in_proj_z, in_proj_a, in_proj_b, dt_bias, A_log, the conv1d's value channels, out_proj's input columns), the
    indexer projection split into q_proj and k_proj, the shared expert gate as a vector and the PLE convolution
    without its middle dimension. The routed experts stay in their GGUF blocks (experts()); the PLE n-gram table too
    (ple_table())."""

    pack_dense = True

    def __init__(self, path: str | Path, text_prefix: str = "model.language_model."):
        files = sorted(Path(path).glob("*.gguf")) if Path(path).is_dir() else [Path(path)]
        if not files:
            raise ValueError(f"no GGUF file in {path}")
        heads = {f: GGUFFile(f) for f in files}
        mtp = [f for f, g in heads.items() if int(g.metadata.get("qwen4exp.nextn_predict_layers", 0) or 0) > 0
               and int(g.metadata.get("split.count", 1)) == 1 and len(g.tensors) < 100]
        main = [f for f in files if f not in mtp]
        first = [f for f in main if "-00001-of-" in f.name] or main
        if not first:
            raise ValueError(f"no GGUF model file in {path} (only MTP layers)")
        self.metadata, self.raw = open_model(first[0])
        if self.metadata.get("general.architecture") != "qwen4exp":
            raise ValueError(f"{first[0]}: GGUF architecture {self.metadata.get('general.architecture')!r} is not "
                             f"supported (qwen4exp is)")
        m = self.metadata
        self.prefix = text_prefix
        self.key_heads = int(m["qwen4exp.ssm.group_count"])
        self.value_heads = int(m["qwen4exp.ssm.time_step_rank"])
        self.key_dim = int(m["qwen4exp.ssm.state_size"])
        self.value_dim = int(m["qwen4exp.ssm.inner_size"]) // self.value_heads
        self.layers = int(m["qwen4exp.block_count"])
        self._cache: dict[str, np.ndarray] = {}
        self.names_map: dict[str, tuple] = {}
        self.expert_map: dict[str, dict[str, Tensor]] = {}
        self.slices: dict[str, tuple[str, int, int, int]] = {}
        p = text_prefix
        for f in mtp[:1]:
            self._add_mtp(heads[f])
        self.names_map[f"{p}embed_tokens.weight"] = ("token_embd.weight",)
        self.names_map["lm_head.weight"] = ("output.weight",)
        for k in ("norm", "down", "up"):
            self.names_map[f"{p}hyper_connection_mixer.{_HC[k]}.weight"] = (f"output_hc_{k}.weight",)
        for raw in self.raw:
            if not raw.startswith("blk."):
                continue
            _, block, suffix = raw.split(".", 2)
            base = f"{p}layers.{block}."
            if suffix in _LAYER:
                self.names_map[base + _LAYER[suffix]] = (raw,)
            elif suffix in _EXPERTS:
                self.expert_map.setdefault(base + "mlp.experts", {})[_EXPERTS[suffix]] = self.raw[raw]
            elif suffix in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
                q, k = f"blk.{block}.indexer.q_proj.weight", f"blk.{block}.indexer.k_proj.weight"
                self.names_map[base + "self_attn.indexer.index_qk_proj.weight"] = (q, k)
        self.meta_tensors: dict[str, np.ndarray] = {}
        for layer in np.asarray(m.get("qwen4exp.ple.layers", []), dtype=np.int64).tolist():
            pe = f"{p}layers.{layer}.ple.ple_embedding."
            for name, key in (("layer_multipliers", "multipliers"), ("ngram_heads_vocab_sizes", "head_vocab_sizes"),
                              ("ngram_heads_offsets", "head_offsets")):
                values = m.get(f"qwen4exp.ple.{key}" if key != "multipliers" else "qwen4exp.ple.layer_multipliers")
                self.meta_tensors[pe + name] = np.asarray(values, dtype=np.int64)
        known = {r for v in self.names_map.values() for r in v} | {r for r, *_ in self.slices.values()} | {
            "per_layer_token_embd.weight"}
        experts = {id(t) for e in self.expert_map.values() for t in e.values()}
        unknown = sorted(r for r, t in self.raw.items() if r not in known and id(t) not in experts)
        if unknown:
            raise ValueError(f"GGUF tensors that seventy does not map: {unknown[:8]}")

    def _add_mtp(self, f: GGUFFile) -> None:
        """The MTP layer of a separate GGUF file (llama.cpp's nextn layer, block num_layers) under the checkpoint's
        mtp.* names. Its own embedding and output head are not used (the drafter uses the model's). eh_proj is
        fc_embedding and fc_hidden side by side (input columns [embedding | hidden]), hc_*_down_inject the mix rows
        and the injection rows of a hyper-connection's first product."""
        block = int(f.metadata["qwen4exp.block_count"]) - 1
        b = f"blk.{block}."
        hidden = int(f.metadata["qwen4exp.embedding_length"])
        rank = int(f.metadata.get("qwen4exp.hyper_connection.low_rank", 0))
        for name, t in f.tensors.items():
            if not name.startswith(b):
                continue
            raw = "mtp:" + name
            self.raw[raw] = t
            suffix = name[len(b):]
            if suffix == "nextn.eh_proj.weight":
                self.slices["mtp.fc_embedding.weight"] = (raw, 1, 0, hidden)
                self.slices["mtp.fc_hidden.weight"] = (raw, 1, hidden, 2 * hidden)
            elif suffix == "nextn.enorm.weight":
                self.names_map["mtp.pre_fc_norm_embedding.weight"] = (raw,)
            elif suffix == "nextn.hnorm.weight":
                self.names_map["mtp.pre_fc_norm_hidden.weight"] = (raw,)
            elif suffix.startswith("nextn.hc_head_"):
                part = suffix[len("nextn.hc_head_") : -len(".weight")]
                self.names_map[f"mtp.hyper_connection_mixer.{_HC[part]}.weight"] = (raw,)
            elif suffix.endswith("_down_inject.weight"):
                branch = "attn_hyper_connection" if suffix.startswith("hc_attn") else "mlp_hyper_connection"
                rows = t.dims[1]
                self.slices[f"mtp.layers.0.{branch}.input_mix_weight_down.weight"] = (raw, 0, 0, rank)
                self.slices[f"mtp.layers.0.{branch}.block_inject_weight.weight"] = (raw, 0, rank, rows)
            elif suffix in _LAYER:
                self.names_map["mtp.layers.0." + _LAYER[suffix]] = (raw,)
            elif suffix in _EXPERTS:
                self.expert_map.setdefault("mtp.layers.0.mlp.experts", {})[_EXPERTS[suffix]] = t
            elif suffix in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
                self.names_map["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"] = (
                    "mtp:" + b + "indexer.q_proj.weight", "mtp:" + b + "indexer.k_proj.weight")
            else:
                raise ValueError(f"GGUF MTP tensor that seventy does not map: {name}")

    def __contains__(self, name: str) -> bool:
        return name in self.names_map or name in self.meta_tensors or name in self.slices

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in (*self.names_map, *self.meta_tensors, *self.slices) if n.startswith(prefix))

    @property
    def text_prefix(self) -> str:
        return self.prefix

    def _untile(self, x: np.ndarray, axis: int, head: int) -> np.ndarray:
        """Undo the tile-major order of the value heads along axis (head values per head)."""
        g, r = self.key_heads, self.value_heads // self.key_heads
        shape = list(x.shape)
        tiled = shape[:axis] + [r, g, head] + shape[axis + 1 :]
        return np.ascontiguousarray(np.swapaxes(x.reshape(tiled), axis, axis + 1)).reshape(shape)

    def _value(self, name: str) -> np.ndarray:
        """The tensor of name in the checkpoint's layout, FP32. The last few stay cached: the loaders read one
        tensor in several row ranges (the tensor-parallel parts of fused projections)."""
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        x = self._restore(name)
        self._cache[name] = x
        while len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))
        return x

    @staticmethod
    def _dequantize(t: Tensor) -> np.ndarray:
        """t's values in the PyTorch shape: on the current GPU for large block-quantized tensors when the kernel
        library is there (FP16, then FP32: the loaders keep FP16 weights), else on the CPU."""
        if t.type not in ("F32", "F16", "BF16") and t.data.size > (1 << 20):
            import torch

            from .ops import kernels, kernels_available

            if torch.cuda.is_available() and kernels_available():
                raw = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
                return kernels.gguf_dequant(raw, t.type, t.dims[0]).float().cpu().numpy().reshape(t.shape)
        return t.dequantize()

    def _restore(self, name: str) -> np.ndarray:
        if name in self.slices:
            raw, axis, a, b = self.slices[name]
            x = self._dequantize(self.raw[raw])
            return np.ascontiguousarray(x[:, a:b] if axis == 1 else x[a:b])
        raws = self.names_map[name]
        x = np.concatenate([self._dequantize(self.raw[r]) for r in raws], axis=0)
        if name.endswith("A_log"):
            x = np.log(-x.astype(np.float64)).astype(np.float32)
        qk = 2 * self.key_heads * self.key_dim
        if name.endswith(("in_proj_z.weight",)):
            x = self._untile(x, 0, self.value_dim)
        elif name.endswith(("in_proj_a.weight", "in_proj_b.weight", "dt_bias", "A_log")):
            x = self._untile(x, 0, 1)
        elif name.endswith(("in_proj_qkv.weight", "linear_attn.conv1d.weight")):
            x = np.concatenate((x[:qk], self._untile(x[qk:], 0, self.value_dim)))
        elif name.endswith("out_proj.weight") and ".linear_attn." in name:
            x = self._untile(x, 1, self.value_dim)
        if (name.endswith(("norm.weight", "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight",
                           "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"))
                and not name.endswith("linear_attn.norm.weight")):
            x = x - 1.0
        if name.endswith("shared_expert_gate.weight") and x.ndim == 1:
            x = x[None]
        if name.endswith("conv1d.weight") and x.ndim == 2:
            x = x[:, None, :]
        return x

    def tensor(self, name: str):
        import torch

        if name in self.meta_tensors:
            return torch.from_numpy(self.meta_tensors[name].copy())
        return torch.from_numpy(np.array(self._value(name)))

    def shape(self, name: str) -> tuple[int, ...]:
        if name in self.slices:
            raw, axis, a, b = self.slices[name]
            full = list(self.raw[raw].shape)
            full[axis] = b - a
            return tuple(full)
        raws = self.names_map[name]
        shapes = [self.raw[r].shape for r in raws]
        return (sum(s[0] for s in shapes), *shapes[0][1:])

    def rows(self, name: str, start: int, stop: int):
        """Rows [start, stop) of a tensor. The embedding and output head rows (a vocabulary shard) are dequantized on
        the current GPU when the kernel library is there (FP16), else on the CPU."""
        import torch

        if name.endswith(("embed_tokens.weight", "lm_head.weight")) and name in self.names_map:
            raws = self.names_map[name]
            t = self.raw[raws[0]]
            if t.type not in ("F32", "F16", "BF16") and torch.cuda.is_available():
                from .ops import kernels, kernels_available

                if kernels_available():
                    raw = torch.from_numpy(np.ascontiguousarray(t.data[start:stop])).cuda()
                    return kernels.gguf_dequant(raw, t.type, t.dims[0]).reshape(stop - start, *t.shape[1:])
            return torch.from_numpy(dequantize(t.data[start:stop], t.type).reshape(stop - start, *t.shape[1:]))
        return torch.from_numpy(np.array(self._value(name)[start:stop]))

    def kind(self, module: str) -> str:
        return "dense"

    def is_nvfp4(self, module: str) -> bool:
        return False

    def out_rows(self, module: str) -> int:
        return self.shape(f"{module}.weight")[0]

    def linear_part(self, module: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None):
        """Rows and columns [start, stop) of a linear module's weight, FP32."""
        w = self.tensor(f"{module}.weight") if rows is None else self.rows(f"{module}.weight", *rows)
        if cols is not None:
            w = w[:, cols[0] : cols[1]]
        return w.contiguous()

    def dense_part(self, name: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None):
        return self.linear_part(name.removesuffix(".weight"), rows, cols)

    def experts(self, module: str) -> dict[str, Tensor]:
        """The routed experts of a MoE module ("...layers.N.mlp.experts"): {"gate", "up", "down": Tensor} with dims
        [k, n, experts] (ggml order), each expert's rows in their GGUF blocks."""
        return self.expert_map[module]

    def ple_table(self) -> Tensor:
        """The PLE n-gram table [rows, dim] in its GGUF blocks."""
        return self.raw["per_layer_token_embd.weight"]

    def close(self) -> None:
        pass

    def __enter__(self) -> "GGUFCheckpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
