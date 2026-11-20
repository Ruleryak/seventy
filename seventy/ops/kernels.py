"""Thin wrappers that run our CUDA kernels on PyTorch tensors. Inputs must be CUDA tensors on the current device."""

from __future__ import annotations

import os

import torch

from ..nvfp4 import GemmWeight, Nvfp4Weight
from ._lib import check, lib


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


def _expect(t: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, not {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _expect(x, "x", torch.float16)
    _expect(weight, "weight", torch.float32)
    dim = x.shape[-1]
    if weight.numel() != dim:
        raise ValueError(f"weight has {weight.numel()} values for rows of {dim}")
    out = torch.empty_like(x) if out is None else out
    _expect(out, "out", torch.float16)
    if residual is not None:
        _expect(residual, "residual", torch.float16)
        residual_out = torch.empty_like(residual) if residual_out is None else residual_out
        _expect(residual_out, "residual_out", torch.float16)
    rows = x.numel() // dim if dim else 0
    code = lib().seventy_rms_norm(
        out.data_ptr(),
        residual_out.data_ptr() if residual is not None else None,
        x.data_ptr(),
        residual.data_ptr() if residual is not None else None,
        weight.data_ptr(),
        rows,
        dim,
        eps,
        _stream(x),
    )
    check("rms_norm", code)
    return out, (residual_out if residual is not None else None)


def nvfp4_dequantize(w: Nvfp4Weight, out: torch.Tensor | None = None) -> torch.Tensor:
    _expect(w.packed, "packed", torch.uint8)
    _expect(w.scales, "scales", torch.uint8)
    n, k = w.shape
    out = torch.empty((n, k), dtype=torch.float16, device=w.packed.device) if out is None else out
    _expect(out, "out", torch.float16)
    code = lib().seventy_nvfp4_dequantize(
        out.data_ptr(), w.packed.data_ptr(), w.scales.data_ptr(), w.global_scale, n, k, _stream(out)
    )
    check("nvfp4_dequantize", code)
    return out


_SMS: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMS:
        _SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SMS[index]


def nvfp4_gemm_config(w: GemmWeight, m: int, device: torch.device) -> tuple[int, int]:
    """(k splits, warps per block) for a weight and batch size, from sweeps on V100 (bench/bench_gemm.py --sweep):
    about 1100 warps in flight up to 16 rows, about 900 above, and at least 4 (up to 16 rows) or 8 k steps per split."""
    tiles = w.n_padded // 32
    ksteps = w.k // 64
    small = m <= 16
    warps = 8 if small else 4
    splits = round((1088 if small else 900) / tiles)
    return max(1, min(splits, max(1, ksteps // (4 if small else 8)))), warps


def nvfp4_gemm(x: torch.Tensor, w: GemmWeight, splits: int | None = None, warps: int | None = None,
               out: torch.Tensor | None = None, workspace: torch.Tensor | None = None) -> torch.Tensor:
    _expect(x, "x", torch.float16)
    m, k = x.shape
    if k != w.k:
        raise ValueError(f"x has k = {k}, the weight has k = {w.k}")
    auto_splits, auto_warps = nvfp4_gemm_config(w, m, x.device)
    splits = auto_splits if splits is None else splits
    warps = auto_warps if warps is None else warps
    need = lib().seventy_nvfp4_gemm_workspace(m, w.n_padded, splits)
    if need and (workspace is None or workspace.numel() * workspace.element_size() < need):
        workspace = torch.empty(need // 4 + 1, dtype=torch.float32, device=x.device)
    out = torch.empty(m, w.n, dtype=torch.float16, device=x.device) if out is None else out
    code = lib().seventy_nvfp4_gemm(
        out.data_ptr(), x.data_ptr(), w.wpk.data_ptr(), w.wsc.data_ptr(), w.alpha.data_ptr(), m, w.n, w.n_padded,
        k, splits, warps, workspace.data_ptr() if need else None,
        workspace.numel() * workspace.element_size() if need else 0, _stream(x),
    )
    check("nvfp4_gemm", code)
    return out


def nvfp4_gemm_dequantize(w: GemmWeight, out: torch.Tensor | None = None) -> torch.Tensor:
    """The dense FP16 weight [n, k] of a repacked weight."""
    out = torch.empty(w.n, w.k, dtype=torch.float16, device=w.wpk.device) if out is None else out
    _expect(out, "out", torch.float16)
    code = lib().seventy_nvfp4_gemm_dequantize(out.data_ptr(), w.wpk.data_ptr(), w.wsc.data_ptr(), w.alpha.data_ptr(),
                                               w.n, w.n_padded, w.k, _stream(out))
    check("nvfp4_gemm_dequantize", code)
    return out


def gdn_conv(y: torch.Tensor, channels: int, conv_w: torch.Tensor, conv_state: torch.Tensor, query_start_t: torch.Tensor,
             context_lens_t: torch.Tensor, state_slots_t: torch.Tensor) -> torch.Tensor:
    """silu(causal depthwise convolution) of the first `channels` columns of y FP16 [T, ld], FP16 [T, channels],
    continuing each sequence from its history in conv_state (updated in place)."""
    _expect(y, "y", torch.float16)
    t = y.shape[0]
    out = torch.empty(t, channels, dtype=torch.float16, device=y.device)
    code = lib().seventy_gdn_conv(out.data_ptr(), conv_state.data_ptr(), y.data_ptr(), y.stride(0), conv_w.data_ptr(),
                                  query_start_t.data_ptr(), context_lens_t.data_ptr(), state_slots_t.data_ptr(),
                                  context_lens_t.numel(), t, channels, conv_w.shape[1], _stream(y))
    check("gdn_conv", code)
    return out


def gdn_gates(y: torch.Tensor, a_off: int, b_off: int, hv: int, a_log: torch.Tensor,
              dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """g = -exp(a_log) * softplus(a + dt_bias) and beta = sigmoid(b), FP32 [T, hv], from columns of y FP16."""
    _expect(y, "y", torch.float16)
    t = y.shape[0]
    g = torch.empty(t, hv, dtype=torch.float32, device=y.device)
    beta = torch.empty_like(g)
    code = lib().seventy_gdn_gates(g.data_ptr(), beta.data_ptr(), y.data_ptr(), y.stride(0), a_off, b_off,
                                   a_log.data_ptr(), dt_bias.data_ptr(), t, hv, _stream(y))
    check("gdn_gates", code)
    return g, beta


def gated_rms_norm(x: torch.Tensor, y: torch.Tensor, z_off: int, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """rms_norm(x) * weight * silu(z) for x FP16 [T, hv, 128], z = columns z_off + h * 128 of y FP16 [T, ld]."""
    _expect(x, "x", torch.float16)
    t, hv, dv = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    code = lib().seventy_gated_rms_norm(out.data_ptr(), x.data_ptr(), y.data_ptr(), y.stride(0), z_off,
                                        weight.data_ptr(), t, hv, dv, eps, _stream(x))
    check("gated_rms_norm", code)
    return out


def gdn_recurrent(
    y: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    query_start: torch.Tensor,
    context_lens: torch.Tensor,
    state_slots: torch.Tensor,
    hk: int,
    hv: int,
    eps: float,
    out: torch.Tensor | None = None,
    commit: bool = True,
    commit_lens: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fused Gated DeltaNet decode/verify step; see seventy_gdn_recurrent in csrc/seventy.h. With commit_lens, the
    commit pass of speculative decoding: no outputs, the state takes the first commit_lens[b] rows."""
    _expect(y, "y", torch.float16)
    _expect(conv_state, "conv_state", torch.float16)
    _expect(state, "state", torch.float32)
    for name, t in (("conv_weight", conv_weight), ("a_log", a_log), ("dt_bias", dt_bias), ("norm_weight", norm_weight)):
        _expect(t, name, torch.float32)
    for name, t in (("query_start", query_start), ("context_lens", context_lens), ("state_slots", state_slots)):
        _expect(t, name, torch.int32)
    dk, dv = state.shape[2], state.shape[3]
    if commit_lens is None:
        out = torch.empty(y.shape[0], hv * dv, dtype=torch.float16, device=y.device) if out is None else out
    else:
        _expect(commit_lens, "commit_lens", torch.int32)
    code = lib().seventy_gdn_recurrent(
        out.data_ptr() if out is not None else None, conv_state.data_ptr(), state.data_ptr(), y.data_ptr(),
        y.shape[1], conv_weight.data_ptr(), a_log.data_ptr(), dt_bias.data_ptr(), norm_weight.data_ptr(),
        query_start.data_ptr(), context_lens.data_ptr(), state_slots.data_ptr(),
        commit_lens.data_ptr() if commit_lens is not None else None, 1 if (commit or commit_lens is not None) else 0,
        context_lens.numel(), hk, hv, dk, dv, conv_weight.shape[1], eps, _stream(y),
    )
    check("gdn_recurrent", code)
    return out


def _kv_args(k_cache, v_cache, k_scale, v_scale):
    if k_cache.dtype == torch.int8:
        if k_scale is None or v_scale is None:
            raise ValueError("int8 KV caches need their scales")
        return k_scale.data_ptr(), v_scale.data_ptr(), 1
    if k_cache.dtype != torch.float16:
        raise ValueError(f"KV caches must be float16 or int8, not {k_cache.dtype}")
    return None, None, 0


def attn_prep(qkv: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor, positions: torch.Tensor,
              inv_freq: torch.Tensor, section: tuple[int, int, int], slots: torch.Tensor, k_cache: torch.Tensor,
              v_cache: torch.Tensor, hq: int, hkv: int, eps: float, k_scale: torch.Tensor | None = None,
              v_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Normed, rotated queries [T, hq, 256]; keys and values written to the caches at `slots` (int32 [T])."""
    _expect(qkv, "qkv", torch.float16)
    _expect(positions, "positions", torch.int64)
    _expect(slots, "slots", torch.int32)
    t = qkv.shape[0]
    q = torch.empty(t, hq, 256, dtype=torch.float16, device=qkv.device)
    ks, vs, kind = _kv_args(k_cache, v_cache, k_scale, v_scale)
    code = lib().seventy_attn_prep(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), ks, vs, kind, qkv.data_ptr(), qkv.shape[1], q_norm.data_ptr(),
        k_norm.data_ptr(), positions.data_ptr(), positions.shape[1], inv_freq.data_ptr(), section[1], section[2],
        2 * inv_freq.numel(), slots.data_ptr(), t, hq, hkv, 256, eps, _stream(qkv),
    )
    check("attn_prep", code)
    return q


def decode_attention_splits(num_seqs: int, hkv: int, device: torch.device) -> int:
    """Context splits per sequence for the decode kernel: about two blocks per SM in total. (The verify kernel runs
    one block per SM, so it takes one split per SM.)"""
    return max(1, min(256, round(2 * _sm_count(device) / max(1, num_seqs * hkv))))


VERIFY_MIN_SPLIT_KEYS = 128


def decode_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     block_tables: torch.Tensor, query_start: torch.Tensor, context_lens: torch.Tensor,
                     max_query_len: int, max_blocks: int, scale: float, k_scale: torch.Tensor | None = None,
                     v_scale: torch.Tensor | None = None, min_split_keys: int | None = None) -> torch.Tensor:
    """Gated attention output [T, hq * 256] for steps with few new tokens per sequence."""
    t, hq, _ = q.shape
    hkv, block_size = k_cache.shape[2], k_cache.shape[1]
    if max_query_len > 1:
        splits = max(1, min(256, round(_sm_count(q.device) / max(1, context_lens.numel() * hkv))))
    else:
        splits = decode_attention_splits(context_lens.numel(), hkv, q.device)
    if max_blocks:
        splits = max(1, min(splits, max_blocks // 2))
    need = lib().seventy_decode_attention_workspace(t, hq, splits)
    ws = torch.empty(need // 4 + 1, dtype=torch.float32, device=q.device)
    out = torch.empty(t, hq * 256, dtype=torch.float16, device=q.device)
    ks, vs, kind = _kv_args(k_cache, v_cache, k_scale, v_scale)
    common = (out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), ks, vs, kind, qkv.data_ptr(),
              qkv.shape[1], block_tables.data_ptr(), block_tables.shape[1], query_start.data_ptr(),
              context_lens.data_ptr(), context_lens.numel(), t, max_query_len, block_size, hq, hkv, 256, splits)
    if max_query_len > 1:
        code = lib().seventy_verify_attention(*common, min_split_keys or VERIFY_MIN_SPLIT_KEYS, scale, ws.data_ptr(),
                                              ws.numel() * 4, _stream(q))
    else:
        code = lib().seventy_decode_attention(*common, scale, ws.data_ptr(), ws.numel() * 4, _stream(q))
    check("decode_attention", code)
    return out


def prefill_tiles(query_start: list[int], device) -> torch.Tensor:
    """(sequence, first new token) of every 64-token query tile, int32 [tiles, 2]."""
    tiles = [(b, q0) for b in range(len(query_start) - 1) for q0 in range(0, query_start[b + 1] - query_start[b], 64)]
    return torch.tensor(tiles, dtype=torch.int32, device=device).view(-1, 2)


GROUPED_ROWS = 48
GROUPED_MIN_BLOCKS = 1024


_last_groups: tuple = (None, None)


def prefill_groups(query_start: list[int], context_lens: list[int], size: int, device) -> tuple[torch.Tensor, int]:
    """Groups of at most `size` new tokens per sequence for seventy_grouped_attention, int32 [3, groups + 1] on the
    device: block-table rows, first token rows (and the end), cached keys before the first token. The last result
    is kept, so the layers of one step copy it to the device once."""
    global _last_groups
    key = (tuple(query_start), tuple(context_lens), size, str(device))
    if _last_groups[0] == key:
        return _last_groups[1]
    rows, start, ctx = [], [0], []
    for b in range(len(query_start) - 1):
        q0, q1 = query_start[b], query_start[b + 1]
        for t in range(q0, q1, size):
            rows.append(b)
            ctx.append(context_lens[b] + t - q0)
            start.append(min(t + size, q1))
    rows.append(0)
    ctx.append(0)
    result = torch.tensor([rows, start, ctx], dtype=torch.int32).to(device), len(rows) - 1
    _last_groups = (key, result)
    return result


def grouped_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      block_tables: torch.Tensor, query_start: list[int], context_lens: list[int], scale: float,
                      k_scale: torch.Tensor, v_scale: torch.Tensor, splits: int | None = None) -> torch.Tensor:
    """Gated causal attention output [T, hq * 256] for prefill steps over INT8 caches, on the verify kernel."""
    t, hq, _ = q.shape
    hkv, block_size = k_cache.shape[2], k_cache.shape[1]
    size = GROUPED_ROWS // (hq // hkv)
    groups, n = prefill_groups(query_start, context_lens, size, q.device)
    if splits is None:
        splits = max(1, min(64, -(-GROUPED_MIN_BLOCKS // (n * hkv))))
    ws = torch.empty(t * hq * splits * 258 + 1, dtype=torch.float32, device=q.device)
    out = torch.empty(t, hq * 256, dtype=torch.float16, device=q.device)
    code = lib().seventy_grouped_attention(
        out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        qkv.data_ptr(), qkv.shape[1], block_tables.data_ptr(), block_tables.shape[1], groups[0].data_ptr(),
        groups[1].data_ptr(), groups[2].data_ptr(), n, t, size, block_size, hq, hkv, 256, splits,
        VERIFY_MIN_SPLIT_KEYS, scale, ws.data_ptr(), ws.numel() * 4, _stream(q),
    )
    check("grouped_attention", code)
    return out


def flash_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_tables: torch.Tensor, query_start: list[int], query_start_t: torch.Tensor,
                    context_lens_t: torch.Tensor, scale: float, k_scale: torch.Tensor | None = None,
                    v_scale: torch.Tensor | None = None, context_lens: list[int] | None = None) -> torch.Tensor:
    """Gated causal attention output [T, hq * 256] for prefill steps. With INT8 caches and the host context lengths,
    it runs on the verify kernel (grouped_attention); SEVENTY_PREFILL_ATTN=tiles keeps the per-head prefill kernel."""
    t, hq, _ = q.shape
    hkv, block_size = k_cache.shape[2], k_cache.shape[1]
    if (k_cache.dtype == torch.int8 and context_lens is not None and hq % hkv == 0 and hq // hkv <= GROUPED_ROWS
            and block_size & (block_size - 1) == 0 and os.environ.get("SEVENTY_PREFILL_ATTN") != "tiles"):
        return grouped_attention(q, qkv, k_cache, v_cache, block_tables, query_start, context_lens, scale, k_scale,
                                 v_scale)
    tiles = prefill_tiles(query_start, q.device)
    out = torch.empty(t, hq * 256, dtype=torch.float16, device=q.device)
    ks, vs, kind = _kv_args(k_cache, v_cache, k_scale, v_scale)
    code = lib().seventy_flash_attention(
        out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), ks, vs, kind, qkv.data_ptr(),
        qkv.shape[1], tiles.data_ptr(), tiles.shape[0], block_tables.data_ptr(), block_tables.shape[1],
        query_start_t.data_ptr(), context_lens_t.data_ptr(), block_size, hq, hkv, 256, scale, _stream(q),
    )
    check("flash_attention", code)
    return out


class DenseWeight:
    """An FP16 weight [n, k] repacked for seventy_dense_gemm: [n_padded / 32, k / 64, 32 lanes, 64 halves]."""

    def __init__(self, w: torch.Tensor):
        if w.dtype != torch.float16 or w.dim() != 2 or w.shape[1] % 64:
            raise ValueError("dense GEMM weights must be FP16 [n, k] with k a multiple of 64")
        n, k = w.shape
        n_padded = -(-n // 32) * 32
        padded = torch.zeros(n_padded, k, dtype=w.dtype, device=w.device)
        padded[:n] = w
        lanes = torch.tensor([8 * ((l >> 2) & 3) + (l & 3) + 4 * (l >= 16) for l in range(32)], device=w.device)
        tiles = padded.view(n_padded // 32, 32, k // 64, 64)[:, lanes].permute(0, 2, 1, 3).contiguous()
        self.wpk, self.n, self.k, self.n_padded = tiles, n, k, n_padded


_LANES = [8 * ((l >> 2) & 3) + (l & 3) + 4 * (l >= 16) for l in range(32)]


class DenseWeightFP8:
    """An FP16 weight [n, k] quantized to FP8 E4M3 with one scale per output row (the speculative drafter only:
    lossy), packed for seventy_dense_gemm_fp8 as [n_padded / 32, k / 64, 32 lanes, 64 bytes] (the DenseWeight
    layout, one byte per value). col_scale: FP32 [n_padded], the row scale times 2^8 (the kernel's E4M3 decoding
    gives value * 2^-8). Rows are quantized in chunks, so a large weight needs little extra memory."""

    def __init__(self, w: torch.Tensor, chunk: int = 8192):
        if w.dtype != torch.float16 or w.dim() != 2 or w.shape[1] % 64:
            raise ValueError("dense GEMM weights must be FP16 [n, k] with k a multiple of 64")
        n, k = w.shape
        n_padded = -(-n // 32) * 32
        q = torch.zeros(n_padded, k, dtype=torch.uint8, device=w.device)
        scale = torch.zeros(n_padded, dtype=torch.float32, device=w.device)
        for r in range(0, n, chunk):
            part = w[r : r + chunk].float()
            s = part.abs().amax(-1) / 448.0
            q[r : r + part.shape[0]] = (part / s.clamp(min=1e-30)[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
            scale[r : r + part.shape[0]] = s
        lanes = torch.tensor(_LANES, device=w.device)
        self.wpk = q.view(n_padded // 32, 32, k // 64, 64)[:, lanes].permute(0, 2, 1, 3).contiguous()
        self.col_scale = scale * 256.0
        self.n, self.k, self.n_padded = n, k, n_padded

    @classmethod
    def from_dense(cls, w: "DenseWeight", chunk_tiles: int = 256) -> "DenseWeightFP8":
        """The FP8 version of a packed FP16 weight (unpacked one group of tiles at a time)."""
        inv = torch.empty(32, dtype=torch.long)
        inv[torch.tensor(_LANES)] = torch.arange(32)
        inv = inv.to(w.wpk.device)
        tiles = w.n_padded // 32
        parts = []
        for t in range(0, tiles, chunk_tiles):
            blk = w.wpk[t : t + chunk_tiles].permute(0, 2, 1, 3)[:, inv]
            parts.append(cls(blk.reshape(-1, w.k)))
        out = cls.__new__(cls)
        out.wpk = torch.cat([p.wpk for p in parts])
        out.col_scale = torch.cat([p.col_scale for p in parts])
        out.n, out.k, out.n_padded = w.n, w.k, w.n_padded
        return out

    def dequantize(self) -> torch.Tensor:
        """FP32 [n, k] of the quantized values (for tests)."""
        inv = torch.empty(32, dtype=torch.long)
        inv[torch.tensor(_LANES)] = torch.arange(32)
        q = self.wpk.permute(0, 2, 1, 3)[:, inv.to(self.wpk.device)].reshape(self.n_padded, self.k)
        vals = q.view(torch.float8_e4m3fn).float() * (self.col_scale / 256.0)[:, None]
        return vals[: self.n]


def dense_gemm(x: torch.Tensor, w: "DenseWeight | DenseWeightFP8", out_fp32: bool = True,
               row_scale: torch.Tensor | None = None) -> torch.Tensor:
    """x [m <= 64, k] (FP16) @ w^T, FP32 accumulation, times row_scale[m] (FP32) when given. FP8 weights
    (DenseWeightFP8) use seventy_dense_gemm_fp8."""
    _expect(x, "x", torch.float16)
    m, k = x.shape
    if k != w.k:
        raise ValueError(f"x has k = {k}, the weight has k = {w.k}")
    if row_scale is not None:
        _expect(row_scale, "row_scale", torch.float32)
    tiles = w.n_padded // 32
    small = m <= 16
    warps = 8 if small else 4
    splits = max(1, min(round((1088 if small else 900) / tiles), max(1, (k // 64) // (4 if small else 8))))
    if not isinstance(w, DenseWeightFP8) and tiles >= 100 and k // 64 >= 8:
        splits = 1
        if tiles >= 1000:
            warps = 8 if small else 2
        elif m > 32 and tiles >= 200:
            splits = max(1, min(round(900 / tiles), max(1, (k // 64) // 8)))
        else:
            warps = 2 if small or tiles >= 200 else 4
    need = lib().seventy_dense_gemm_workspace(m, w.n_padded, splits)
    ws = torch.empty(need // 4 + 1, dtype=torch.float32, device=x.device) if need else None
    out = torch.empty(m, w.n, dtype=torch.float32 if out_fp32 else torch.float16, device=x.device)
    rs = row_scale.data_ptr() if row_scale is not None else None
    wsp = ws.data_ptr() if ws is not None else None
    if isinstance(w, DenseWeightFP8):
        code = lib().seventy_dense_gemm_fp8(out.data_ptr(), 1 if out_fp32 else 0, x.data_ptr(), w.wpk.data_ptr(),
                                            w.col_scale.data_ptr(), m, w.n, w.n_padded, k, splits, warps, wsp, need,
                                            rs, _stream(x))
        check("dense_gemm_fp8", code)
        return out
    code = lib().seventy_dense_gemm(out.data_ptr(), 1 if out_fp32 else 0, x.data_ptr(), w.wpk.data_ptr(), m, w.n,
                                    w.n_padded, k, splits, warps, wsp, need, rs, _stream(x))
    check("dense_gemm", code)
    return out


def drafter_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                 residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """RMS norm of x (+ residual, FP32) with an FP32 weight: (FP16 output, the new FP32 residual or None)."""
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("drafter_norm takes FP16 or FP32 input")
    x = x.contiguous()
    rows, dim = x.shape
    out = torch.empty(rows, dim, dtype=torch.float16, device=x.device)
    res_out = torch.empty(rows, dim, dtype=torch.float32, device=x.device) if residual is not None else None
    if residual is not None:
        _expect(residual, "residual", torch.float32)
    code = lib().seventy_drafter_norm(out.data_ptr(), res_out.data_ptr() if res_out is not None else None,
                                      x.data_ptr(), 1 if x.dtype == torch.float32 else 0,
                                      residual.data_ptr() if residual is not None else None, weight.data_ptr(),
                                      rows, dim, eps, _stream(x))
    check("drafter_norm", code)
    return out, res_out


def drafter_conv(x: torch.Tensor, base: torch.Tensor, delta: torch.Tensor, group_size: int, block: int,
                 out_fp32: bool) -> torch.Tensor:
    """The two-tap grouped convolution (csrc/drafter.cu). base: FP32 [2, channels]; delta: FP32 [rows, 2, groups]
    (a row-strided view is fine)."""
    x = x.contiguous()
    rows, channels = x.shape
    groups = delta.shape[-1]
    if delta.stride(-1) != 1 or delta.stride(-2) != groups or delta.dtype != torch.float32:
        raise ValueError("delta must be FP32 [rows, 2, groups] with contiguous groups")
    base = base.contiguous()
    out = torch.empty(rows, channels, dtype=torch.float32 if out_fp32 else torch.float16, device=x.device)
    code = lib().seventy_drafter_conv(out.data_ptr(), 1 if out_fp32 else 0, x.data_ptr(),
                                      1 if x.dtype == torch.float32 else 0, base.data_ptr(), delta.data_ptr(),
                                      delta.stride(0), rows, groups, group_size, block, _stream(x))
    check("drafter_conv", code)
    return out


def drafter_qk(q: torch.Tensor | None, kv: torch.Tensor, q_norm: torch.Tensor | None, k_norm: torch.Tensor,
               positions: torch.Tensor, slots: torch.Tensor, inv_freq: torch.Tensor, k_ring: torch.Tensor,
               v_ring: torch.Tensor, heads: int, kv_heads: int, eps: float) -> torch.Tensor | None:
    """Norm and rotary for the query heads (returned, FP16 [rows, heads, 128]) and the key heads, and K and V into
    the layer's rings [slots, ring, kv_heads, 128] at position % ring. q: FP32 [rows, heads * 128]; kv: FP32
    [rows, 2 * kv_heads * 128]; both may be row-strided views."""
    rows = kv.shape[0]
    d = 128
    ring = k_ring.shape[1]
    if kv.stride(-1) != 1 or (q is not None and q.stride(-1) != 1):
        raise ValueError("q and kv rows must be contiguous")
    positions = positions.to(torch.int64).contiguous()
    slots = slots.to(torch.int64).contiguous()
    q_out = torch.empty(rows, heads, d, dtype=torch.float16, device=kv.device) if q is not None else None
    code = lib().seventy_drafter_qk(q_out.data_ptr() if q_out is not None else None, k_ring.data_ptr(),
                                    v_ring.data_ptr(), q.data_ptr() if q is not None else None,
                                    q.stride(0) if q is not None else 0, kv.data_ptr(), kv.stride(0),
                                    q_norm.data_ptr() if q_norm is not None else None, k_norm.data_ptr(),
                                    positions.data_ptr(), slots.data_ptr(), inv_freq.data_ptr(), rows,
                                    heads if q is not None else 0, kv_heads, d, ring, eps, _stream(kv))
    check("drafter_qk", code)
    return q_out


def drafter_attention(q: torch.Tensor, k_ring: torch.Tensor, v_ring: torch.Tensor, slots: torch.Tensor,
                      starts: torch.Tensor, block: int, window: int, scale: float) -> torch.Tensor:
    """Block attention of the drafter over its rings: q FP16 [seqs * block, heads, 128] -> FP16
    [seqs * block, heads * 128]."""
    _expect(q, "q", torch.float16)
    rows, heads, d = q.shape
    seqs = rows // block
    kv_heads, ring = k_ring.shape[2], k_ring.shape[1]
    slots = slots.to(torch.int64).contiguous()
    starts = starts.to(torch.int64).contiguous()
    splits = lib().seventy_drafter_attention_splits(block, window)
    need = lib().seventy_drafter_attention_workspace(seqs, kv_heads, splits)
    ws = torch.empty(need // 4 + 1, dtype=torch.float32, device=q.device)
    out = torch.empty(rows, heads * d, dtype=torch.float16, device=q.device)
    code = lib().seventy_drafter_attention(out.data_ptr(), q.data_ptr(), k_ring.data_ptr(), v_ring.data_ptr(),
                                           slots.data_ptr(), starts.data_ptr(), seqs, heads, kv_heads, d, block,
                                           window, ring, scale, ws.data_ptr(), need, _stream(q))
    check("drafter_attention", code)
    return out


def silu_mul_scale(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """silu(gate) * up of FP32 [rows, 2 * inter], scaled per row into the FP16 range: (FP16 [rows, inter], FP32
    scale [rows])."""
    _expect(gate_up, "gate_up", torch.float32)
    gate_up = gate_up.contiguous()
    rows, two = gate_up.shape
    out = torch.empty(rows, two // 2, dtype=torch.float16, device=gate_up.device)
    scale = torch.empty(rows, dtype=torch.float32, device=gate_up.device)
    check("silu_mul_scale", lib().seventy_silu_mul_scale(out.data_ptr(), scale.data_ptr(), gate_up.data_ptr(), rows,
                                                         two // 2, _stream(gate_up)))
    return out, scale


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up of FP16 [rows, 2 * inter] = [gate | up] -> FP16 [rows, inter]."""
    _expect(gate_up, "gate_up", torch.float16)
    rows, two = gate_up.shape
    out = torch.empty(rows, two // 2, dtype=torch.float16, device=gate_up.device)
    check("silu_mul", lib().seventy_silu_mul(out.data_ptr(), gate_up.data_ptr(), rows, two // 2, _stream(gate_up)))
    return out
