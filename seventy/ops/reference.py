"""PyTorch reference implementations. They define what each kernel must compute, and they run every operation
on CPU tensors. The math is FP32 with the same rounding points as the kernels.

Varlen operations take the batch layout of a model step: sequence b owns token rows
[query_start[b], query_start[b + 1]), and its cache already holds context_lens[b] tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .. import nvfp4
from ..nvfp4 import Nvfp4Weight


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    s = x if residual is None else (x.float() + residual.float()).to(x.dtype)
    f = s.float()
    out = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return out.to(x.dtype), (None if residual is None else s)


def gated_rms_norm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """rms_norm(x) * weight * silu(z), over the last dimension."""
    f = x.float()
    out = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * weight.float() * F.silu(z.float())
    return out.to(x.dtype)


def nvfp4_dequantize(w: Nvfp4Weight) -> torch.Tensor:
    return nvfp4.dequantize(w, torch.float16)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up, where gate_up holds [gate | up] along the last dimension."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(gate_up.dtype)


def sigmoid_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.sigmoid(gate.float())).to(x.dtype)


def rotary_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, section: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin [T, rotary_dim / 2] for positions [3, T] (temporal, height, width).

    Interleaved multimodal rotary: frequency i takes the height position if i % 3 == 1 and i < 3 * section[1],
    the width position if i % 3 == 2 and i < 3 * section[2], and the temporal position otherwise.
    """
    half = inv_freq.numel()
    axis = torch.zeros(half, dtype=torch.long)
    idx = torch.arange(half)
    axis[(idx % 3 == 1) & (idx < 3 * section[1])] = 1
    axis[(idx % 3 == 2) & (idx < 3 * section[2])] = 2
    pos = positions.to(torch.float32)[axis.to(positions.device)].T
    angles = pos * inv_freq.to(device=positions.device, dtype=torch.float32)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first rotary_dim values of each head of x [T, H, D]: value j pairs with value j + rotary_dim / 2."""
    half = cos.shape[-1]
    f = x.float()
    a, b, rest = f[..., :half], f[..., half : 2 * half], f[..., 2 * half :]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat((a * c - b * s, b * c + a * s, rest), dim=-1).to(x.dtype)


def kv_slots(
    query_start: list[int], context_lens: list[int], block_tables: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Cache slot (block * block_size + offset) of every new token of the step, [T]."""
    slots = []
    for b in range(len(context_lens)):
        n = query_start[b + 1] - query_start[b]
        pos = torch.arange(context_lens[b], context_lens[b] + n, device=block_tables.device)
        slots.append(block_tables[b, pos // block_size].long() * block_size + pos % block_size)
    return torch.cat(slots) if slots else torch.empty(0, dtype=torch.long, device=block_tables.device)


def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 values and FP32 scales of the vectors in the last dimension: scale = max |x| / 127, q = round(x / scale)."""
    f = x.float()
    scale = f.abs().amax(-1) / 127.0
    q = torch.where(scale[..., None] > 0, torch.round(f / scale[..., None].clamp(min=1e-38)), torch.zeros_like(f))
    return q.clamp(-127, 127).to(torch.int8), scale


def write_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor,
             k_scale: torch.Tensor | None = None, v_scale: torch.Tensor | None = None):
    """Store k and v [T, Hkv, D] in the paged caches [blocks, block_size, Hkv, D]. int8 caches also take the
    scales [blocks, block_size, Hkv]."""
    if k_cache.dtype == torch.int8:
        for cache, scales, x in ((k_cache, k_scale, k), (v_cache, v_scale, v)):
            q, s = quantize_int8(x)
            cache.view(-1, *cache.shape[2:])[slots] = q
            scales.view(-1, scales.shape[-1])[slots] = s
        return
    k_cache.view(-1, *k_cache.shape[2:])[slots] = k.to(k_cache.dtype)
    v_cache.view(-1, *v_cache.shape[2:])[slots] = v.to(v_cache.dtype)


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    query_start: list[int],
    context_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal attention of q [T, Hq, D] over each sequence's cached tokens, which include the new tokens."""
    block_size = k_cache.shape[1]
    hq, hkv = q.shape[1], k_cache.shape[2]
    out = torch.empty_like(q)
    for b in range(len(context_lens)):
        s, e = query_start[b], query_start[b + 1]
        n = e - s
        if n == 0:
            continue
        total = context_lens[b] + n
        pos = torch.arange(total, device=q.device)
        slots = block_tables[b, pos // block_size].long() * block_size + pos % block_size
        k = k_cache.view(-1, hkv, k_cache.shape[-1])[slots].float()
        v = v_cache.view(-1, hkv, v_cache.shape[-1])[slots].float()
        if k_cache.dtype == torch.int8:
            k = k * k_scale.view(-1, hkv)[slots][..., None]
            v = v * v_scale.view(-1, hkv)[slots][..., None]
        k = k.repeat_interleave(hq // hkv, dim=1)
        v = v.repeat_interleave(hq // hkv, dim=1)
        scores = torch.einsum("thd,shd->hts", q[s:e].float(), k) * scale
        qpos = torch.arange(context_lens[b], total, device=q.device)
        scores.masked_fill_(pos[None, None, :] > qpos[None, :, None], float("-inf"))
        out[s:e] = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), v).to(q.dtype)
    return out


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start: list[int],
    context_lens: list[int],
    state_slots: list[int],
) -> torch.Tensor:
    """Depthwise causal convolution over time with SiLU, per sequence, continuing from the cached inputs.

    x: [T, C]. weight: [C, K]. conv_state: [slots, C, K - 1], the last K - 1 inputs of each sequence (zero for a
    sequence without context). Output: [T, C]. conv_state is updated in place.
    """
    width = weight.shape[1]
    out = torch.empty_like(x)
    w = weight.float()
    for b in range(len(context_lens)):
        s, e = query_start[b], query_start[b + 1]
        slot = state_slots[b]
        prev = conv_state[slot].float() if context_lens[b] > 0 else x.new_zeros(x.shape[1], width - 1).float()
        seq = torch.cat((prev, x[s:e].float().T), dim=1)
        y = F.conv1d(seq.unsqueeze(0), w.unsqueeze(1), groups=x.shape[1]).squeeze(0)
        out[s:e] = F.silu(y).T.to(x.dtype)
        conv_state[slot] = seq[:, seq.shape[1] - (width - 1) :].to(conv_state.dtype)
    return out


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log decay g = -exp(A_log) * softplus(a + dt_bias) and write strength beta = sigmoid(b), both FP32 [T, Hv]."""
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    return g, torch.sigmoid(b.float())


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _prepare_qk(q: torch.Tensor, k: torch.Tensor, hv: int) -> tuple[torch.Tensor, torch.Tensor]:
    rep = hv // q.shape[1]
    q = _l2norm(q.float()) * (1.0 / math.sqrt(q.shape[-1]))
    k = _l2norm(k.float())
    return q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)


def gated_delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    query_start: list[int],
    context_lens: list[int],
    state_slots: list[int],
) -> torch.Tensor:
    """The gated delta rule, one token at a time. This is the definition the chunked form must match.

    q, k: [T, Hk, Dk]; v: [T, Hv, Dv]; g, beta: [T, Hv]; state: [slots, Hv, Dk, Dv] FP32, updated in place.
    q and k are L2-normalized per head, and q is scaled by 1 / sqrt(Dk). Value head h reads key head
    h // (Hv / Hk). For each token: S = exp(g) * S; S += k (beta * (v - S^T k))^T; o = S^T q.
    """
    hv = v.shape[1]
    qf, kf = _prepare_qk(q, k, hv)
    vf = v.float()
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for b in range(len(context_lens)):
        s, e = query_start[b], query_start[b + 1]
        slot = state_slots[b]
        S = state[slot].clone() if context_lens[b] > 0 else state.new_zeros(state.shape[1:])
        for t in range(s, e):
            S = S * torch.exp(g[t])[:, None, None]
            kv = torch.einsum("hkv,hk->hv", S, kf[t])
            delta = (vf[t] - kv) * beta[t][:, None]
            S = S + kf[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hkv,hk->hv", S, qf[t]).to(v.dtype)
        state[slot] = S
    return out


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    query_start: list[int],
    context_lens: list[int],
    state_slots: list[int],
    chunk: int = 64,
) -> torch.Tensor:
    """The gated delta rule in chunks: the same result as gated_delta_rule_recurrent, with matrix products.

    Within a chunk with cumulative log decay G and state S0 at its start, the corrected values
    U = T (B V) - T (B Gamma K) S0 with T = (I + A)^-1, A[t, j] = beta_t exp(G_t - G_j) k_t.k_j for j < t,
    B = diag(beta), Gamma = diag(exp(G)). Then O = (Gamma Q) S0 + (M * Q K^T) U with M[t, j] = exp(G_t - G_j)
    for j <= t, and the state after the chunk is exp(G_last) S0 + K^T diag(exp(G_last - G)) U.
    """
    hv = v.shape[1]
    qf, kf = _prepare_qk(q, k, hv)
    vf = v.float()
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for b in range(len(context_lens)):
        s, e = query_start[b], query_start[b + 1]
        slot = state_slots[b]
        S = state[slot].clone() if context_lens[b] > 0 else state.new_zeros(state.shape[1:])
        for c0 in range(s, e, chunk):
            c1 = min(c0 + chunk, e)
            n = c1 - c0
            Q = qf[c0:c1].transpose(0, 1)
            K = kf[c0:c1].transpose(0, 1)
            V = vf[c0:c1].transpose(0, 1)
            G = torch.cumsum(g[c0:c1].T, dim=-1)
            bt = beta[c0:c1].T
            diff = G[:, :, None] - G[:, None, :]
            lower = torch.tril(torch.ones(n, n, dtype=torch.bool, device=q.device))
            decay = torch.where(lower, torch.exp(diff.masked_fill(~lower, 0.0)), torch.zeros_like(diff))
            strict = torch.tril(torch.ones(n, n, device=q.device), diagonal=-1)
            A = bt[:, :, None] * decay * (K @ K.transpose(1, 2)) * strict
            eye = torch.eye(n, device=q.device).expand(hv, n, n)
            rhs = torch.cat((bt[:, :, None] * V, (bt * torch.exp(G))[:, :, None] * K), dim=-1)
            sol = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
            U = sol[..., : V.shape[-1]] - sol[..., V.shape[-1] :] @ S
            O = (torch.exp(G)[:, :, None] * Q) @ S + (decay * (Q @ K.transpose(1, 2))) @ U
            out[c0:c1] = O.transpose(0, 1).to(v.dtype)
            last = G[:, -1:]
            S = torch.exp(last)[:, :, None] * S + (K * torch.exp(last - G)[:, :, None]).transpose(1, 2) @ U
        state[slot] = S
    return out
