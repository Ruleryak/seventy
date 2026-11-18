"""Gated DeltaNet prefill in chunks of 64 tokens.

For a chunk with cumulative log decay G (within the chunk), write strengths beta, normalized keys K, scaled
normalized queries Q, values V, and state S at the chunk start:

- A[t, j] = beta_t exp(G_t - G_j) k_t.k_j for j < t; T = (I + A)^-1.
- U' = T (beta V), W = T (beta exp(G) K): the corrected values are U = U' - W S.
- O = (exp(G) Q) S + P U, with P[t, j] = exp(G_t - G_j) q_t.k_j for j <= t.
- The next state is exp(G_last) S + (K exp(G_last - G))^T U.

Everything that does not depend on S is computed here for all chunks at once (FP32, batched); the sequential
part runs in the gdn_chunk_scan kernel.
"""

from __future__ import annotations

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


def gated_delta_rule(q, k, v, g, beta, state, query_start, query_start_t, context_lens_t, state_slots_t):
    """Chunked gated delta rule on the GPU: output FP16 [T, Hv, Dv]; the state slots are updated in place."""
    t, hv, d = v.shape
    prep = prepare(q, k, v, g, beta, query_start)
    out = torch.empty(t, hv, d, dtype=torch.float16, device=v.device)
    code = kernels.lib().seventy_gdn_chunk_scan(
        out.data_ptr(), state.data_ptr(), prep["w"].data_ptr(), prep["u"].data_ptr(), prep["gq"].data_ptr(),
        prep["p"].data_ptr(), prep["klt"].data_ptr(), prep["eg"].data_ptr(), prep["chunk_start"].data_ptr(),
        query_start_t.data_ptr(), context_lens_t.data_ptr(), state_slots_t.data_ptr(), len(query_start) - 1, hv,
        prep["chunks"], kernels._stream(v),
    )
    kernels.check("gdn_chunk_scan", code)
    return out
