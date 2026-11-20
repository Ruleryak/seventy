"""Gated DeltaNet prefill in chunks of 64 tokens.

For a chunk with cumulative log decay G (within the chunk), write strengths beta, normalized keys K, scaled
normalized queries Q, values V, and state S at the chunk start:

- A[t, j] = beta_t exp(G_t - G_j) k_t.k_j for j < t; T = (I + A)^-1.
- U' = T (beta V), W = T (beta exp(G) K): the corrected values are U = U' - W S.
- O = (exp(G) Q) S + P U, with P[t, j] = exp(G_t - G_j) q_t.k_j for j <= t.
- The next state is exp(G_last) S + (K exp(G_last - G))^T U.

Everything that does not depend on S is computed for all chunks at once (FP32): by the gdn_prepare kernel when q,
k and v are the column blocks of one convolution output (the model's layout), else by prepare() here in PyTorch
(the reference for the kernel). The sequential part runs in seventy_gdn_chunk_scan: the state recurrence alone,
4 blocks per value head, then the outputs of all chunks in parallel (SEVENTY_GDN_SCAN=single: one kernel, one
block per value head, as before).
"""

from __future__ import annotations

import os

import torch

from . import kernels
from .reference import _prepare_qk

CHUNK = 64


def _chunk_index(query_start: list[int], device) -> tuple[torch.Tensor, list[int]]:
    """Token row of every chunk position [chunks, 64] (-1 for padding), and each sequence's first chunk."""
    rows, starts = [], []
    for b in range(len(query_start) - 1):
        s, e = query_start[b], query_start[b + 1]
        starts.append(len(rows))
        for c0 in range(s, e, CHUNK):
            rows.append([t if t < e else -1 for t in range(c0, c0 + CHUNK)])
    return torch.tensor(rows, dtype=torch.long, device=device).view(-1, CHUNK), starts


def prepare(q, k, v, g, beta, query_start: list[int], heads_per_group: int = 8):
    """Per-chunk inputs of gdn_chunk_scan, as FP16 [Hv, chunks, ...] tensors (FP32 for exp(G_last))."""
    t, hv, d = v.shape
    idx, starts = _chunk_index(query_start, v.device)
    nc = idx.shape[0]
    pad = idx.clamp(min=0)
    valid = (idx >= 0).float()
    out = {
        "w": torch.empty(hv, nc, CHUNK, d, dtype=torch.float16, device=v.device),
        "u": torch.empty(hv, nc, CHUNK, d, dtype=torch.float16, device=v.device),
        "gq": torch.empty(hv, nc, CHUNK, d, dtype=torch.float16, device=v.device),
        "p": torch.empty(hv, nc, CHUNK, CHUNK, dtype=torch.float16, device=v.device),
        "klt": torch.empty(hv, nc, d, CHUNK, dtype=torch.float16, device=v.device),
        "eg": torch.empty(hv, nc, dtype=torch.float32, device=v.device),
    }
    qf, kf = _prepare_qk(q, k, hv)
    lower = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=v.device))
    strict = torch.tril(torch.ones(CHUNK, CHUNK, device=v.device), diagonal=-1)
    eye = torch.eye(CHUNK, device=v.device)
    for h0 in range(0, hv, heads_per_group):
        h1 = min(hv, h0 + heads_per_group)

        def gather(x):
            return (x[:, h0:h1][pad] * valid[..., None, None]).permute(2, 0, 1, 3)

        Q, K, V = gather(qf), gather(kf), gather(v.float())
        G = torch.cumsum((g[:, h0:h1][pad] * valid[..., None]).permute(2, 0, 1), dim=-1)
        bt = (beta[:, h0:h1][pad] * valid[..., None]).permute(2, 0, 1)
        diff = G[..., :, None] - G[..., None, :]
        decay = torch.where(lower, torch.exp(diff.masked_fill(~lower, 0.0)), torch.zeros_like(diff))
        A = bt[..., :, None] * decay * (K @ K.transpose(-1, -2)) * strict
        rhs = torch.cat((bt[..., None] * V, (bt * torch.exp(G))[..., None] * K), dim=-1)
        X = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        last = G[..., -1:]
        out["u"][h0:h1] = X[..., :d].half()
        out["w"][h0:h1] = X[..., d:].half()
        out["gq"][h0:h1] = (torch.exp(G)[..., None] * Q).half()
        out["p"][h0:h1] = (decay * (Q @ K.transpose(-1, -2))).half()
        out["klt"][h0:h1] = (K * torch.exp(last - G)[..., None]).transpose(-1, -2).half()
        out["eg"][h0:h1] = torch.exp(last[..., 0])
    out["chunk_start"] = torch.tensor(starts, dtype=torch.int32, device=v.device)
    out["chunks"] = nc
    return out


def _fused_layout(q, k, v, g, beta) -> bool:
    """True when q, k, v [T, H, 128] are the q | k | v column blocks of one FP16 buffer with an even row stride,
    and g, beta are contiguous FP32 [T, Hv]: the layout gdn_prepare reads."""
    t, hk, d = q.shape
    hv = v.shape[1]
    ld = q.stride(0)
    return (q.is_cuda and q.dtype == k.dtype == v.dtype == torch.float16 and d == 128 and v.shape[2] == 128
            and q.stride()[1:] == (d, 1) and k.stride() == q.stride() and v.stride() == (ld, d, 1) and ld % 2 == 0
            and k.data_ptr() == q.data_ptr() + hk * d * 2 and v.data_ptr() == q.data_ptr() + 2 * hk * d * 2
            and g.dtype == beta.dtype == torch.float32 and g.is_contiguous() and beta.is_contiguous()
            and tuple(g.shape) == tuple(beta.shape) == (t, hv) and os.environ.get("SEVENTY_GDN_PREPARE") != "torch")


def prepare_fused(q, k, v, g, beta, query_start: list[int]):
    """prepare() by the gdn_prepare kernel (q, k, v in the layout _fused_layout checks)."""
    t, hk, d = q.shape
    hv = v.shape[1]
    rows, lens, starts = [], [], []
    for b in range(len(query_start) - 1):
        s, e = query_start[b], query_start[b + 1]
        starts.append(len(rows))
        for c0 in range(s, e, CHUNK):
            rows.append(c0)
            lens.append(min(CHUNK, e - c0))
    nc = len(rows)
    dev = v.device
    meta = torch.tensor(rows + lens, dtype=torch.int32).to(dev, non_blocking=True)
    out = {
        "w": torch.empty(hv, nc, CHUNK, d, dtype=torch.float16, device=dev),
        "u": torch.empty(hv, nc, CHUNK, d, dtype=torch.float16, device=dev),
        "gq": torch.empty(hv, nc, CHUNK, d, dtype=torch.float16, device=dev),
        "p": torch.empty(hv, nc, CHUNK, CHUNK, dtype=torch.float16, device=dev),
        "klt": torch.empty(hv, nc, d, CHUNK, dtype=torch.float16, device=dev),
        "eg": torch.empty(hv, nc, dtype=torch.float32, device=dev),
    }
    code = kernels.lib().seventy_gdn_prepare(
        out["w"].data_ptr(), out["u"].data_ptr(), out["gq"].data_ptr(), out["p"].data_ptr(), out["klt"].data_ptr(),
        out["eg"].data_ptr(), q.data_ptr(), q.stride(0), g.data_ptr(), beta.data_ptr(), meta.data_ptr(),
        meta.data_ptr() + nc * 4, nc, hk, hv, d, v.shape[2], kernels._stream(v))
    kernels.check("gdn_prepare", code)
    out["chunk_start"] = torch.tensor(starts, dtype=torch.int32).to(dev, non_blocking=True)
    out["chunks"] = nc
    return out


def gated_delta_rule(q, k, v, g, beta, state, query_start, query_start_t, context_lens_t, state_slots_t):
    """Chunked gated delta rule on the GPU: output FP16 [T, Hv, Dv]; the state slots are updated in place."""
    t, hv, d = v.shape
    if _fused_layout(q, k, v, g, beta):
        prep = prepare_fused(q, k, v, g, beta, query_start)
    else:
        prep = prepare(q, k, v, g, beta, query_start)
    out = torch.empty(t, hv, d, dtype=torch.float16, device=v.device)
    split = os.environ.get("SEVENTY_GDN_SCAN") != "single"
    nc = prep["chunks"]
    hs = torch.empty(hv, nc, d, d, dtype=torch.float16, device=v.device) if split else None
    vt = torch.empty(hv, nc, d, CHUNK, dtype=torch.float16, device=v.device) if split else None
    code = kernels.lib().seventy_gdn_chunk_scan(
        out.data_ptr(), state.data_ptr(), prep["w"].data_ptr(), prep["u"].data_ptr(), prep["gq"].data_ptr(),
        prep["p"].data_ptr(), prep["klt"].data_ptr(), prep["eg"].data_ptr(), prep["chunk_start"].data_ptr(),
        query_start_t.data_ptr(), context_lens_t.data_ptr(), state_slots_t.data_ptr(), len(query_start) - 1, hv,
        nc, hs.data_ptr() if split else None, vt.data_ptr() if split else None, kernels._stream(v),
    )
    kernels.check("gdn_chunk_scan", code)
    return out
