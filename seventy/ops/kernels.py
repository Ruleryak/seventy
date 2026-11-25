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


def gated_rms_norm(x: torch.Tensor, y: torch.Tensor, z_off: int, weight: torch.Tensor, eps: float,
                   sigmoid_gate: bool = False) -> torch.Tensor:
    """rms_norm(x) * weight * silu(z) (sigmoid(z) with sigmoid_gate) for x FP16 [T, hv, 128], z = columns
    z_off + h * 128 of y FP16 [T, ld]."""
    _expect(x, "x", torch.float16)
    t, hv, dv = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    code = lib().seventy_gated_rms_norm(out.data_ptr(), x.data_ptr(), y.data_ptr(), y.stride(0), z_off,
                                        weight.data_ptr(), t, hv, dv, eps, int(sigmoid_gate), _stream(x))
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
    sigmoid_gate: bool = False,
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
        context_lens.numel(), hk, hv, dk, dv, conv_weight.shape[1], eps, int(sigmoid_gate), _stream(y),
    )
    check("gdn_recurrent", code)
    return out


def gdn_commit_layers(
    y_all: torch.Tensor,
    conv_all: torch.Tensor,
    state_all: torch.Tensor,
    conv_weight_all: torch.Tensor,
    a_log_all: torch.Tensor,
    dt_bias_all: torch.Tensor,
    norm_all: torch.Tensor,
    query_start: torch.Tensor,
    context_lens: torch.Tensor,
    state_slots: torch.Tensor,
    commit_lens: torch.Tensor,
    hk: int,
    hv: int,
    eps: float,
) -> None:
    """The commit pass of gdn_recurrent for the first y_all.shape[0] Gated DeltaNet layers in one launch; see
    seventy_gdn_commit_layers in csrc/seventy.h. y_all: [layers, T, ld]; conv_all, state_all: the caches of all
    layers; the parameters stacked per layer: conv weights [layers, conv_dim, taps], a_log and dt_bias
    [layers, hv], norm [layers, dv]."""
    _expect(y_all, "y_all", torch.float16)
    _expect(conv_all, "conv_all", torch.float16)
    _expect(state_all, "state_all", torch.float32)
    for name, x in (("conv_weight_all", conv_weight_all), ("a_log_all", a_log_all), ("dt_bias_all", dt_bias_all),
                    ("norm_all", norm_all)):
        _expect(x, name, torch.float32)
    for name, x in (("query_start", query_start), ("context_lens", context_lens), ("state_slots", state_slots),
                    ("commit_lens", commit_lens)):
        _expect(x, name, torch.int32)
    layers, tokens, ld = y_all.shape
    dk, dv = state_all.shape[3], state_all.shape[4]
    if (conv_all.shape[0] < layers or state_all.shape[0] < layers or conv_weight_all.shape[0] < layers
            or a_log_all.shape != (layers, hv) or dt_bias_all.shape != (layers, hv) or norm_all.shape != (layers, dv)):
        raise ValueError("gdn_commit_layers: per-layer arrays do not match the layer count")
    code = lib().seventy_gdn_commit_layers(
        conv_all.data_ptr(), state_all.data_ptr(), y_all.data_ptr(), ld, tokens * ld, conv_all.stride(0),
        state_all.stride(0), conv_weight_all.data_ptr(), a_log_all.data_ptr(), dt_bias_all.data_ptr(),
        norm_all.data_ptr(), query_start.data_ptr(), context_lens.data_ptr(), state_slots.data_ptr(),
        commit_lens.data_ptr(), context_lens.numel(), layers, hk, hv, dk, dv, conv_weight_all.shape[2], eps,
        _stream(y_all),
    )
    check("gdn_commit_layers", code)


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


def topk(x: torch.Tensor, k: int, id_offset: int = 0, ids: torch.Tensor | None = None
         ) -> tuple[torch.Tensor, torch.Tensor]:
    """(values FP32 [rows, k], ids int32 [rows, k]) of the k <= 64 largest values of each row of FP32 x [rows, n],
    by value descending and then id ascending (the first is the argmax with the smallest id). The id of element e is
    ids[row, e] (int32, negative ids skipped) or id_offset + e."""
    rows, n = x.shape
    if x.dtype != torch.float32 or x.stride(1) != 1 or (ids is not None and (ids.dtype != torch.int32
                                                                              or ids.stride(1) != 1)):
        raise ValueError("topk takes FP32 rows (and int32 ids) with unit column stride")
    values = torch.empty(rows, k, dtype=torch.float32, device=x.device)
    out_ids = torch.empty(rows, k, dtype=torch.int32, device=x.device)
    need = lib().seventy_topk_workspace(rows, n, k)
    ws = torch.empty(max(1, need // 4), dtype=torch.float32, device=x.device) if need else None
    code = lib().seventy_topk(values.data_ptr(), out_ids.data_ptr(), x.data_ptr(), x.stride(0),
                              ids.data_ptr() if ids is not None else None, ids.stride(0) if ids is not None else 0,
                              rows, n, k, id_offset, ws.data_ptr() if ws is not None else None, need, _stream(x))
    check("topk", code)
    return values, out_ids


def spec_accept(pred: torch.Tensor, vals: torch.Tensor, idx: torch.Tensor, drafts: torch.Tensor, q: torch.Tensor,
                cand: torch.Tensor, temp: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor, min_p: torch.Tensor,
                u_accept: torch.Tensor, u_residual: torch.Tensor, stop: torch.Tensor, limit: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """device_accept_top followed by cut_at_stops (seventy/engine/spec.py) in one kernel: (accepted drafts after the
    cut int64 [B], new tokens int64 [B, k + 1])."""
    b, k1, top = vals.shape
    k = k1 - 1
    c = q.shape[-1]
    if drafts.stride(1) != 1:
        drafts = drafts.contiguous()
    n = torch.empty(b, dtype=torch.int64, device=vals.device)
    new = torch.empty(b, k1, dtype=torch.int64, device=vals.device)
    t = [pred.contiguous(), vals.contiguous(), idx.contiguous()]
    rest = [q.float().contiguous(), cand.contiguous(), temp.float().contiguous(), top_k.contiguous(),
            top_p.float().contiguous(), min_p.float().contiguous(), u_accept.float().contiguous(),
            u_residual.float().contiguous()]
    st, lim = stop.contiguous(), limit.contiguous()
    code = lib().seventy_spec_accept(n.data_ptr(), new.data_ptr(), *(x.data_ptr() for x in t), drafts.data_ptr(),
                                     drafts.stride(0), *(x.data_ptr() for x in rest), st.data_ptr(), st.shape[1],
                                     lim.data_ptr(), b, k, top, c, _stream(vals))
    check("spec_accept", code)
    return n, new


def drafter_walk(anchors: torch.Tensor, values: torch.Tensor, ids: torch.Tensor, hid: torch.Tensor,
                 pred: torch.Tensor, succ: torch.Tensor, temps: torch.Tensor, gumbel: torch.Tensor | None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """The DFlash2 selector walk (Drafter.walk) in one kernel: (tokens int64 [b, n], probs FP32 [b, n, k])."""
    b, n, k = ids.shape
    rank = pred.shape[1]
    tokens = torch.empty(b, n, dtype=torch.int64, device=ids.device)
    probs = torch.empty(b, n, k, dtype=torch.float32, device=ids.device)
    args = [anchors.contiguous(), values.float().contiguous(), ids.contiguous(), hid.float().contiguous(),
            pred.contiguous(), succ.contiguous(), temps.float().contiguous()]
    g = gumbel.float().contiguous() if gumbel is not None else None
    code = lib().seventy_drafter_walk(tokens.data_ptr(), probs.data_ptr(), *(a.data_ptr() for a in args),
                                      g.data_ptr() if g is not None else None, b, n, k, rank, _stream(ids))
    check("drafter_walk", code)
    return tokens, probs


_last_tiles: tuple = (None, None)


def prefill_tiles(query_start: list[int], device) -> torch.Tensor:
    """(sequence, first new token) of every 64-token query tile, int32 [tiles, 2]. The last result is kept, so the
    layers of one step copy it to the device once."""
    global _last_tiles
    key = (tuple(query_start), str(device))
    if _last_tiles[0] == key:
        return _last_tiles[1]
    tiles = [(b, q0) for b in range(len(query_start) - 1) for q0 in range(0, query_start[b + 1] - query_start[b], 64)]
    out = torch.tensor(tiles, dtype=torch.int32, device=device).view(-1, 2)
    _last_tiles = (key, out)
    return out


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


PREFILL_MIN_SPLIT_TILES = 8
PREFILL_GROUPED_MAX_TOKENS = 4
PREFILL_SPLIT_OVERHEAD = 8.0


def prefill_splits(tiles: int, keys: int, device) -> int:
    """Key splits for seventy_prefill_attention: the count s that minimizes waves(tiles * s blocks) * (overhead +
    key tiles per split), with at least PREFILL_MIN_SPLIT_TILES key tiles of 32 per split. The overhead (Q staging,
    the epilogue and the merge, in key tiles) is PREFILL_SPLIT_OVERHEAD."""
    sms = _sm_count(device)
    key_tiles = -(-keys // 32)
    best, best_cost = 1, None
    for s in range(1, max(1, min(256, key_tiles // PREFILL_MIN_SPLIT_TILES)) + 1):
        cost = -(-tiles * s // sms) * (PREFILL_SPLIT_OVERHEAD + key_tiles / s)
        if best_cost is None or cost < best_cost:
            best, best_cost = s, cost
    return best


_last_prefill_tiles: tuple = (None, None)


def prefill_attention_tiles(query_start: list[int], hq: int, hkv: int, device) -> torch.Tensor:
    """The tiles of seventy_prefill_attention, int32 [tiles, 4] {sequence, first token, first head, heads << 8 |
    tokens}: 64 tokens of one head, and for the last tokens of a sequence (fewer than 64) the packed tiles of
    64 // (hq / hkv) tokens times the hq / hkv heads of a KV head, if they need fewer blocks. The last result is
    kept, so the layers of one step copy it to the device once."""
    global _last_prefill_tiles
    key = (tuple(query_start), hq, hkv, str(device))
    if _last_prefill_tiles[0] == key:
        return _last_prefill_tiles[1]
    g = hq // hkv
    per = 64 // g
    tiles = []
    for b in range(len(query_start) - 1):
        n = query_start[b + 1] - query_start[b]
        full = n // 64
        tiles += [(b, 64 * i, h, 1 << 8 | 64) for i in range(full) for h in range(hq)]
        rest, q0 = n - 64 * full, 64 * full
        if rest == 0:
            continue
        packed = -(-rest // per)
        if packed * hkv < hq:
            tiles += [(b, q0 + per * i, kh * g, g << 8 | min(per, rest - per * i)) for i in range(packed)
                      for kh in range(hkv)]
        else:
            tiles += [(b, q0, h, 1 << 8 | rest) for h in range(hq)]
    out = torch.tensor(tiles, dtype=torch.int32, device=device).view(-1, 4)
    _last_prefill_tiles = (key, out)
    return out


def prefill_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                      block_tables: torch.Tensor, query_start: list[int], query_start_t: torch.Tensor,
                      context_lens_t: torch.Tensor, context_lens: list[int], scale: float, k_scale: torch.Tensor,
                      v_scale: torch.Tensor, splits: int | None = None) -> torch.Tensor:
    """Gated causal attention output [T, hq * 256] for prefill steps over INT8 caches, on the split-D kernel, with
    the keys split as prefill_splits chooses."""
    t, hq, _ = q.shape
    hkv, block_size = k_cache.shape[2], k_cache.shape[1]
    tiles = prefill_attention_tiles(query_start, hq, hkv, q.device)
    n = tiles.shape[0]
    if splits is None:
        keys = max(c + query_start[b + 1] - query_start[b] for b, c in enumerate(context_lens))
        splits = prefill_splits(n, keys, q.device)
    ws = torch.empty(t * hq * splits * 258 if splits > 1 else 1, dtype=torch.float32, device=q.device)
    out = torch.empty(t, hq * 256, dtype=torch.float16, device=q.device)
    code = lib().seventy_prefill_attention(
        out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        qkv.data_ptr(), qkv.shape[1], tiles.data_ptr(), n, block_tables.data_ptr(), block_tables.shape[1],
        query_start_t.data_ptr(), context_lens_t.data_ptr(), t, block_size, hq, hkv, 256, splits, 1, 0, 64, scale,
        ws.data_ptr() if splits > 1 else None, ws.numel() * 4 if splits > 1 else 0, _stream(q),
    )
    check("prefill_attention", code)
    return out


VERIFY_SPLITD = os.environ.get("SEVENTY_VERIFY_ATTN", "splitd") == "splitd"
VERIFY_MIN_SPLIT_TILES = 4
VERIFY_SPLITD_CONTEXT = int(os.environ.get("SEVENTY_VERIFY_SPLITD_CONTEXT", "32768"))
_verify_tiles: dict = {}


def verify_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     block_tables: torch.Tensor, query_start_t: torch.Tensor, context_lens_t: torch.Tensor,
                     tokens: int, scale: float, k_scale: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
    """Gated causal attention [T, hq * 256] over INT8 caches for steps with the same number of new tokens per
    sequence, at most 48 rows per KV head (speculative verify, and decode steps of one token per sequence, where it
    is 4 to 6 times faster than the decode kernel from 2K to 100K): seventy_verify_attention_hybrid, which runs the
    sequences with at least VERIFY_SPLITD_CONTEXT cached keys on the split-D kernel and the others on the verify
    kernel, with a fixed number of key splits (about one block per SM), so the launch suits CUDA graphs. The tiles
    of a shape are made once and kept (graphs hold their addresses). In the engine (TP4, 4 streams) the split-D
    kernel is 13% slower at 8K, even at 32K and 4 to 7% faster from 78K to 200K."""
    t, hq, _ = q.shape
    hkv, block_size = k_cache.shape[2], k_cache.shape[1]
    seqs = context_lens_t.numel()
    key = (seqs, tokens, hq, hkv, str(q.device))
    tiles = _verify_tiles.get(key)
    if tiles is None:
        tiles = _verify_tiles[key] = prefill_attention_tiles([i * tokens for i in range(seqs + 1)], hq, hkv,
                                                             q.device).clone()
    n = tiles.shape[0]
    splits = max(2, min(128, round(_sm_count(q.device) / max(1, n))))
    ws = torch.empty(t * hq * splits * 258, dtype=torch.float32, device=q.device)
    out = torch.empty(t, hq * 256, dtype=torch.float16, device=q.device)
    code = lib().seventy_verify_attention_hybrid(
        out.data_ptr(), q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        qkv.data_ptr(), qkv.shape[1], tiles.data_ptr(), n, block_tables.data_ptr(), block_tables.shape[1],
        query_start_t.data_ptr(), context_lens_t.data_ptr(), seqs, t, tokens, block_size, hq, hkv, splits,
        VERIFY_MIN_SPLIT_TILES, 2 if splits >= 40 else 0, VERIFY_MIN_SPLIT_KEYS, VERIFY_SPLITD_CONTEXT, scale,
        ws.data_ptr(), ws.numel() * 4, _stream(q),
    )
    check("verify_attention", code)
    return out


def flash_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_tables: torch.Tensor, query_start: list[int], query_start_t: torch.Tensor,
                    context_lens_t: torch.Tensor, scale: float, k_scale: torch.Tensor | None = None,
                    v_scale: torch.Tensor | None = None, context_lens: list[int] | None = None) -> torch.Tensor:
    """Gated causal attention output [T, hq * 256] for prefill steps. With INT8 caches and the host context lengths,
    it runs on the split-D kernel (prefill_attention), or on the verify kernel (grouped_attention) when no sequence
    has more than PREFILL_GROUPED_MAX_TOKENS new tokens. SEVENTY_PREFILL_ATTN=grouped always takes the verify kernel,
    SEVENTY_PREFILL_ATTN=tiles the per-head prefill kernel."""
    t, hq, _ = q.shape
    hkv, block_size = k_cache.shape[2], k_cache.shape[1]
    mode = os.environ.get("SEVENTY_PREFILL_ATTN")
    fast = (k_cache.dtype == torch.int8 and context_lens is not None and hq % hkv == 0
            and block_size & (block_size - 1) == 0 and mode not in ("tiles", "wmma"))
    tiny = max(b - a for a, b in zip(query_start, query_start[1:])) <= PREFILL_GROUPED_MAX_TOKENS
    if fast and mode != "grouped" and not (tiny and hq // hkv <= GROUPED_ROWS):
        return prefill_attention(q, qkv, k_cache, v_cache, block_tables, query_start, query_start_t, context_lens_t,
                                 context_lens, scale, k_scale, v_scale)
    if fast and hq // hkv <= GROUPED_ROWS:
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


def dense_unpack(w: DenseWeight) -> torch.Tensor:
    """The FP16 weight [n, k] of a packed DenseWeight."""
    inv = torch.empty(32, dtype=torch.long)
    inv[torch.tensor(_LANES)] = torch.arange(32)
    rows = w.wpk.permute(0, 2, 1, 3)[:, inv.to(w.wpk.device)].reshape(w.n_padded, w.k)
    return rows[: w.n]


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
    def from_fp8(cls, values: torch.Tensor, row_scale: torch.Tensor) -> "DenseWeightFP8":
        """Pack FP8 E4M3 weights as stored ([n, k] bytes, uint8 or float8_e4m3fn) with one FP32 scale per row
        (W = value * row_scale): no rounding, so FP8 checkpoints keep their exact weights."""
        values = values.view(torch.uint8)
        n, k = values.shape
        if k % 64:
            raise ValueError("dense GEMM weights must have k a multiple of 64")
        n_padded = -(-n // 32) * 32
        q = torch.zeros(n_padded, k, dtype=torch.uint8, device=values.device)
        q[:n] = values
        scale = torch.zeros(n_padded, dtype=torch.float32, device=values.device)
        scale[:n] = row_scale.float().to(values.device)
        lanes = torch.tensor(_LANES, device=values.device)
        out = cls.__new__(cls)
        out.wpk = q.view(n_padded // 32, 32, k // 64, 64)[:, lanes].permute(0, 2, 1, 3).contiguous()
        out.col_scale = scale * 256.0
        out.n, out.k, out.n_padded = n, k, n_padded
        return out

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


FP8_WK = os.environ.get("SEVENTY_DRAFTER_FP8_WK", "1") != "0"
_COUNTERS: dict = {}


def _tile_counters(device: torch.device, tiles: int) -> torch.Tensor:
    """The per-tile counters of seventy_dense_gemm_fp8_wk on a device: zero between calls, shared by all calls on
    the device (they run one after another on one stream)."""
    if tiles > 1 << 16:
        raise ValueError(f"dense_gemm_fp8_wk: {tiles} column tiles, at most {1 << 16}")
    c = _COUNTERS.get(device)
    if c is None:
        c = _COUNTERS[device] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return c


def fp8_wk_config(m: int, tiles: int, ksteps: int) -> tuple[int, int] | None:
    """(warps per block, k splits) for seventy_dense_gemm_fp8_wk, or None where the block-staged kernel
    (seventy_dense_gemm_fp8) is faster: from 32 rows up on wide weights, where the warp-split kernel reads the
    activations once per 32-column tile. From sweeps of the drafter's shapes on V100 (lab fp8_wk_bench)."""
    if m > 16 and (ksteps >= 200 or tiles >= 1000 or (m > 32 and tiles >= 200)):
        return None
    if tiles <= 64 and m > 16 and ksteps >= 12:
        return 4, 3
    return (8 if tiles <= 100 else 4 if tiles <= 200 else 2), 1


def dense_gemm(x: torch.Tensor, w: "DenseWeight | DenseWeightFP8", out_fp32: bool = True,
               row_scale: torch.Tensor | None = None, fp8_config: tuple[int, int] | None = None) -> torch.Tensor:
    """x [m <= 64, k] (FP16) @ w^T, FP32 accumulation, times row_scale[m] (FP32) when given. FP8 weights
    (DenseWeightFP8) use seventy_dense_gemm_fp8_wk with (warps, splits) from fp8_wk_config or fp8_config."""
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
    out = torch.empty(m, w.n, dtype=torch.float32 if out_fp32 else torch.float16, device=x.device)
    rs = row_scale.data_ptr() if row_scale is not None else None
    wk = fp8_config or (fp8_wk_config(m, tiles, k // 64) if isinstance(w, DenseWeightFP8) and FP8_WK else None)
    if isinstance(w, DenseWeightFP8) and wk is not None:
        warps, splits = wk
        need = lib().seventy_dense_gemm_fp8_wk_workspace(m, w.n_padded, splits)
        ws = torch.empty(need // 4 + 1, dtype=torch.float32, device=x.device) if need else None
        code = lib().seventy_dense_gemm_fp8_wk(out.data_ptr(), 1 if out_fp32 else 0, x.data_ptr(), w.wpk.data_ptr(),
                                               w.col_scale.data_ptr(), m, w.n, w.n_padded, k, splits, warps,
                                               ws.data_ptr() if ws is not None else None, need,
                                               _tile_counters(x.device, tiles).data_ptr(), rs, _stream(x))
        check("dense_gemm_fp8_wk", code)
        return out
    need = lib().seventy_dense_gemm_workspace(m, w.n_padded, splits)
    ws = torch.empty(need // 4 + 1, dtype=torch.float32, device=x.device) if need else None
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
    splits = lib().seventy_drafter_attention_splits(seqs, kv_heads, block, window)
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


MOE_ROWS_PER_SLOT = int(os.environ.get("SEVENTY_MOE_ROWS", "2"))


def moe_max_slots(pairs: int, experts: int) -> int:
    """Slots of a plan of `pairs` pairs over `experts` local experts: whole slots of MOE_ROWS_PER_SLOT pairs, plus
    one partial slot per expert."""
    return max(1, -(-pairs // MOE_ROWS_PER_SLOT) + min(experts, pairs))


def moe_route(logits: torch.Tensor, k: int, renorm: bool, experts: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(expert ids int32 [tokens, k], weights FP32 [tokens, k]) of router logits FP32 [tokens, >= experts] (the first
    experts columns, default all; rows may be strided): the top k (ties: the lower id), weighted by the softmax over
    the k picked logits (renorm) or over all logits."""
    _expect(logits, "logits", torch.float32)
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    t = logits.shape[0]
    e = logits.shape[1] if experts is None else experts
    ids = torch.empty(t, k, dtype=torch.int32, device=logits.device)
    weights = torch.empty(t, k, dtype=torch.float32, device=logits.device)
    check("moe_route", lib().seventy_moe_route(logits.data_ptr(), logits.stride(0), t, e, k, int(renorm),
                                               ids.data_ptr(), weights.data_ptr(), _stream(logits)))
    return ids, weights


def moe_finish(y: torch.Tensor, ids: torch.Tensor, k: int, e0: int, e1: int, shared: torch.Tensor | None,
               gate: torch.Tensor | None) -> torch.Tensor:
    """The MoE output FP16 [tokens, dim] (seventy_moe_finish): sigmoid(gate[t]) * shared[t] (FP16 [tokens, dim];
    gate FP32 [tokens], may be strided) plus the rows of y (FP32 [tokens * k, dim]) of each token's picks of the
    local experts [e0, e1) (ids int32 [tokens * k])."""
    _expect(y, "y", torch.float32)
    _expect(ids, "ids", torch.int32)
    pairs, dim = y.shape
    t = pairs // k
    out = torch.empty(t, dim, dtype=torch.float16, device=y.device)
    if shared is not None:
        _expect(shared, "shared", torch.float16)
        shared = shared.contiguous()
    check("moe_finish", lib().seventy_moe_finish(
        out.data_ptr(), y.data_ptr(), ids.contiguous().data_ptr(), t, k, e0, e1,
        shared.data_ptr() if shared is not None else None, gate.data_ptr() if gate is not None else None,
        gate.stride(0) if gate is not None else 0, dim, _stream(y)))
    return out


def moe_plan(ids: torch.Tensor, e0: int, e1: int) -> tuple:
    """The pairs of the local experts [e0, e1) in ids (int32 [pairs] expert ids), sorted by expert, and their slots:
    (order, slot_expert, slot_start, slot_count, num_slots, max_slots), all on the GPU (csrc/moe.cu)."""
    _expect(ids, "ids", torch.int32)
    pairs = ids.numel()
    max_slots = moe_max_slots(pairs, e1 - e0)
    dev = ids.device
    order = torch.empty(max(pairs, 1), dtype=torch.int32, device=dev)
    slots = torch.empty(3, max_slots, dtype=torch.int32, device=dev)
    num = torch.empty(1, dtype=torch.int32, device=dev)
    check("moe_plan", lib().seventy_moe_plan(ids.contiguous().data_ptr(), pairs, e0, e1, MOE_ROWS_PER_SLOT, max_slots,
                                             order.data_ptr(), slots[0].data_ptr(), slots[1].data_ptr(),
                                             slots[2].data_ptr(), num.data_ptr(), _stream(ids)))
    return order, slots[0], slots[1], slots[2], num, max_slots


def moe_gemm(x: torch.Tensor, row_div: int, weight: torch.Tensor, scales: torch.Tensor | None,
             alpha: torch.Tensor | None, plan: tuple, pairs: int, pair_weight: torch.Tensor | None = None,
             out: torch.Tensor | None = None) -> torch.Tensor:
    """For every pair of the plan: x[pair / row_div] times its expert's weight (NVFP4 packed [E, n, k / 2] with
    scales [E, n, k / 16] and alpha [E, n], or FP16 [E, n, k]). FP16 [pairs, n], or FP32 [pairs, n] scaled by
    pair_weight. Rows of pairs outside the plan are left as they are in out (zeros when out is new)."""
    _expect(x, "x", torch.float16)
    x = x.contiguous()
    nvfp4 = weight.dtype == torch.uint8
    n = weight.shape[1]
    k = weight.shape[2] * (2 if nvfp4 else 1)
    if out is None:
        out = torch.zeros(pairs, n, dtype=torch.float32 if pair_weight is not None else torch.float16, device=x.device)
    order, slot_expert, slot_start, slot_count, num, max_slots = plan
    code = lib().seventy_moe_gemm(out.data_ptr(), x.data_ptr(), row_div, weight.data_ptr(),
                                  scales.data_ptr() if nvfp4 else None, alpha.data_ptr() if nvfp4 else None,
                                  int(nvfp4), n, k, order.data_ptr(), slot_expert.data_ptr(), slot_start.data_ptr(),
                                  slot_count.data_ptr(), num.data_ptr(), max_slots, MOE_ROWS_PER_SLOT,
                                  pair_weight.data_ptr() if pair_weight is not None else None, _stream(x))
    check("moe_gemm", code)
    return out


def hc_norm(h: torch.Tensor, weight: torch.Tensor, hc: int, eps: float) -> torch.Tensor:
    """RMS norm of each of the hc streams of h FP32 [rows, hc * hidden], times weight FP32 [hc * hidden]: FP16."""
    _expect(h, "h", torch.float32)
    h = h.contiguous()
    rows = h.shape[0]
    out = torch.empty(h.shape, dtype=torch.float16, device=h.device)
    check("hc_norm", lib().seventy_hc_norm(out.data_ptr(), h.data_ptr(), weight.data_ptr(), rows, hc,
                                           h.shape[1] // hc, eps, _stream(h)))
    return out


def hc_act(d: torch.Tensor, rank: int, hc: int, inject: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(silu(d[:, :rank] / hc) FP16, 2 sigmoid(d[:, rank:rank + hc] / hc) FP32 or None) of d FP32 [rows, >= rank]."""
    _expect(d, "d", torch.float32)
    d = d.contiguous()
    rows = d.shape[0]
    s = torch.empty(rows, rank, dtype=torch.float16, device=d.device)
    w = torch.empty(rows, hc, dtype=torch.float32, device=d.device) if inject else None
    check("hc_act", lib().seventy_hc_act(s.data_ptr(), w.data_ptr() if w is not None else None, d.data_ptr(), rows,
                                         rank, hc, d.shape[1], _stream(d)))
    return s, w


def hc_combine(u: torch.Tensor, n: torch.Tensor, hc: int) -> torch.Tensor:
    """mean over streams of sigmoid(u) * n (u FP32, n FP16 [rows, hc * hidden]): FP16 [rows, hidden]."""
    _expect(u, "u", torch.float32)
    _expect(n, "n", torch.float16)
    rows, width = n.shape
    x = torch.empty(rows, width // hc, dtype=torch.float16, device=n.device)
    check("hc_combine", lib().seventy_hc_combine(x.data_ptr(), u.contiguous().data_ptr(), n.contiguous().data_ptr(),
                                                 rows, hc, width // hc, _stream(n)))
    return x


def hc_inject(h: torch.Tensor, y: torch.Tensor, w: torch.Tensor, hc: int) -> torch.Tensor:
    """h FP32 [rows, hc * hidden] += y FP16 [rows, hidden] times w FP32 [rows, hc] per stream, in place."""
    _expect(h, "h", torch.float32)
    _expect(y, "y", torch.float16)
    rows = h.shape[0]
    check("hc_inject", lib().seventy_hc_inject(h.data_ptr(), y.contiguous().data_ptr(), w.contiguous().data_ptr(), rows,
                                               hc, h.shape[1] // hc, _stream(h)))
    return h


def qsa_pool(pooled: torch.Tensor, idx_keys: torch.Tensor, positions: torch.Tensor, block_tables: torch.Tensor,
             query_start_t: torch.Tensor, context_lens_t: torch.Tensor, tokens: int, k_norm: torch.Tensor,
             inv_freq: torch.Tensor, section: tuple[int, int, int], rotary_dim: int, compress: int,
             eps: float) -> None:
    """The pooled keys of the blocks that the step's new tokens complete (seventy_qsa_pool)."""
    bt = block_tables.contiguous()
    check("qsa_pool", lib().seventy_qsa_pool(
        pooled.data_ptr(), idx_keys.data_ptr(), positions.data_ptr(), bt.data_ptr(), bt.shape[1],
        query_start_t.data_ptr(), context_lens_t.data_ptr(), context_lens_t.numel(), tokens, k_norm.data_ptr(),
        inv_freq.data_ptr(), section[1], section[2], rotary_dim, compress, idx_keys.shape[1], eps,
        _stream(idx_keys)))


def qsa_index_prep(qk: torch.Tensor, q_norm: torch.Tensor, idx_cache: torch.Tensor, kv_slots_t: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor, heads: int, eps: float) -> torch.Tensor:
    """Indexer queries FP16 [tokens, heads, 128] (RMS norm times q_norm, rotated with cos/sin [tokens, half]) from
    the projection qk FP32 [tokens, (heads + 1) * 128]; writes the raw keys (the last 128 values) as FP16 into
    idx_cache at kv_slots_t (seventy_qsa_index_prep)."""
    _expect(qk, "qk", torch.float32)
    t = qk.shape[0]
    dim = idx_cache.shape[-1]
    out = torch.empty(t, heads, dim, dtype=torch.float16, device=qk.device)
    half = cos.shape[-1]
    cos, sin = cos.float().contiguous(), sin.float().contiguous()
    check("qsa_index_prep", lib().seventy_qsa_index_prep(
        out.data_ptr(), idx_cache.data_ptr(), qk.data_ptr(), qk.stride(0), q_norm.data_ptr(), kv_slots_t.data_ptr(),
        cos.data_ptr(), sin.data_ptr(), t, heads, dim, half, eps, _stream(qk)))
    return out


def qsa_select(q: torch.Tensor, pooled: torch.Tensor, block_tables: torch.Tensor, query_start_t: torch.Tensor,
               context_lens_t: torch.Tensor, compress: int, top: int, max_blocks: int,
               block_size: int, row0: int = 0) -> torch.Tensor:
    """Positions int32 [rows, top * compress + compress - 1] (-1: none) that each query row attends to
    (seventy_qsa_select), from the indexer queries q FP16 [rows, heads, dim] of the step's rows row0 ..
    row0 + rows - 1."""
    _expect(q, "q", torch.float16)
    q = q.contiguous()
    rows, heads, _ = q.shape
    width = top * compress + compress - 1
    sel = torch.empty(rows, width, dtype=torch.int32, device=q.device)
    ld = max(1, max_blocks)
    scores = torch.empty(rows, ld, dtype=torch.float32, device=q.device)
    bt = block_tables.contiguous()
    check("qsa_select", lib().seventy_qsa_select(
        sel.data_ptr(), width, scores.data_ptr(), ld, q.data_ptr(), heads, pooled.data_ptr(), bt.data_ptr(),
        bt.shape[1], query_start_t.data_ptr(), context_lens_t.data_ptr(), context_lens_t.numel(), row0, rows, compress,
        top, block_size, _stream(q)))
    return sel


QSA_SPLIT_POSITIONS = 32
QSA_BLOCKS = 160


def qsa_attention(q: torch.Tensor, qkv: torch.Tensor, k_cache, v_cache, k_scale, v_scale, sel: torch.Tensor,
                  block_tables: torch.Tensor, query_start_t: torch.Tensor, scale: float,
                  splits: int | None = None, row0: int = 0) -> torch.Tensor:
    """Attention of each query row of q FP16 [rows, hq, 256] over its selected positions sel, times sigmoid of the
    gate in qkv: FP16 [rows, hq * 256] (seventy_qsa_attention, then seventy_attention_combine). By default a row's
    selection splits into parts of QSA_SPLIT_POSITIONS entries while the launch has fewer than QSA_BLOCKS blocks."""
    _expect(q, "q", torch.float16)
    rows, hq, d = q.shape
    hkv = k_cache.shape[2]
    if splits is None:
        splits = max(1, min(-(-sel.shape[1] // QSA_SPLIT_POSITIONS), -(-QSA_BLOCKS // (rows * hkv))))
    ws = torch.empty(rows * hq * splits * (d + 2), dtype=torch.float32, device=q.device)
    po = ws[: rows * hq * splits * d]
    pml = ws[rows * hq * splits * d :]
    bt = block_tables.contiguous()
    int8 = k_cache.dtype == torch.int8
    check("qsa_attention", lib().seventy_qsa_attention(
        po.data_ptr(), pml.data_ptr(), q.contiguous().data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        k_scale.data_ptr() if int8 else None, v_scale.data_ptr() if int8 else None, sel.data_ptr(), sel.shape[1],
        bt.data_ptr(), bt.shape[1], query_start_t.data_ptr(), query_start_t.numel() - 1, row0, rows, k_cache.shape[1],
        hq, hkv, splits, scale, _stream(q)))
    out = torch.empty(rows, hq * d, dtype=torch.float16, device=q.device)
    check("attention_combine", lib().seventy_attention_combine(out.data_ptr(), ws.data_ptr(), qkv.data_ptr(),
                                                               qkv.stride(0), rows, hq, splits, _stream(q)))
    return out


def ple_ids(tokens: torch.Tensor, state: torch.Tensor, slots_t: torch.Tensor, query_start_t: torch.Tensor,
            context_lens_t: torch.Tensor, n: int, heads_per: int, mult: torch.Tensor, sizes: torch.Tensor,
            offsets: torch.Tensor, eos: int) -> torch.Tensor:
    """Table rows int64 [tokens, (n - 1) * heads_per] of the step's tokens (seventy_ple_ids)."""
    t = tokens.numel()
    ids = torch.empty(t, (n - 1) * heads_per, dtype=torch.int64, device=tokens.device)
    check("ple_ids", lib().seventy_ple_ids(ids.data_ptr(), tokens.contiguous().data_ptr(), state.data_ptr(),
                                           slots_t.data_ptr(), query_start_t.data_ptr(), context_lens_t.data_ptr(),
                                           context_lens_t.numel(), t, n, heads_per, mult.data_ptr(), sizes.data_ptr(),
                                           offsets.data_ptr(), eos, _stream(tokens)))
    return ids


def ple_gather(ids: torch.Tensor, h0: int, h1: int, packed: torch.Tensor, scales: torch.Tensor, row_base: int,
               dim: int, global_scale: float) -> torch.Tensor:
    """FP16 [tokens, (h1 - h0) * dim] rows of heads h0 .. h1 - 1 from NVFP4 tables in pinned host memory."""
    t = ids.shape[0]
    out = torch.empty(t, (h1 - h0) * dim, dtype=torch.float16, device=ids.device)
    check("ple_gather", lib().seventy_ple_gather(out.data_ptr(), ids.data_ptr(), ids.shape[1], h0, h1, t,
                                                 packed.data_ptr(), scales.data_ptr(), row_base, dim, global_scale,
                                                 _stream(ids)))
    return out


def ple_conv(x: torch.Tensor, xn: torch.Tensor, state: torch.Tensor, w: torch.Tensor, slots_t: torch.Tensor,
             query_start_t: torch.Tensor, context_lens_t: torch.Tensor, taps: int, dil: int) -> torch.Tensor:
    """x + silu(dilated causal convolution of xn) per sequence (seventy_ple_conv), FP32."""
    out = torch.empty_like(x)
    check("ple_conv", lib().seventy_ple_conv(out.data_ptr(), x.contiguous().data_ptr(), xn.contiguous().data_ptr(),
                                             state.data_ptr(), w.data_ptr(), slots_t.data_ptr(),
                                             query_start_t.data_ptr(), context_lens_t.data_ptr(),
                                             context_lens_t.numel(), x.shape[0], x.shape[1], taps, dil, _stream(x)))
    return out


def ple_state(tok_state: torch.Tensor, conv_state: torch.Tensor, tokens: torch.Tensor, xn: torch.Tensor,
              slots_t: torch.Tensor, query_start_t: torch.Tensor, context_lens_t: torch.Tensor, eos: int) -> None:
    """The token and convolution states of the sequences after the step (seventy_ple_state)."""
    n1 = tok_state.shape[1]
    hist, channels = conv_state.shape[1], conv_state.shape[2]
    check("ple_state", lib().seventy_ple_state(tok_state.data_ptr(), conv_state.data_ptr(),
                                               tokens.contiguous().data_ptr(), xn.contiguous().data_ptr(),
                                               slots_t.data_ptr(), query_start_t.data_ptr(), context_lens_t.data_ptr(),
                                               context_lens_t.numel(), n1, channels, hist, eos, _stream(xn)))

