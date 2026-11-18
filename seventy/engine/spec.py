"""Acceptance of drafted tokens (speculative decoding).

The target verifies [anchor, d1 .. dk] in one step and produces k + 1 distributions: row i is the target's
distribution for the token after position i. Greedy requests accept the longest prefix of drafts that equals the
target's argmax, then take the target's next argmax. Sampled requests use standard rejection sampling, which keeps
the target's distribution exactly: draft d_i is accepted with probability min(1, p_i(d_i) / q_i(d_i)); on the first
rejection the next token is drawn from max(0, p_i - q_i), normalized; if all drafts pass, from p_k. Here q_i is the
drafter's distribution over its candidates at position i (zero elsewhere), and p_i is the target's distribution
after the request's penalties, temperature and filters.
"""

from __future__ import annotations

import torch

from .sampling import RowState, _apply_penalties, _filter


def _plain(r: RowState) -> bool:
    p = r.params
    return (p.temperature == 0 and p.repetition_penalty == 1.0 and not p.presence_penalty
            and not p.frequency_penalty and len(r.output) >= p.min_tokens)


def _target_probs(logits: torch.Tensor, row: RowState, eos_ids: tuple[int, ...]) -> torch.Tensor:
    p = row.params
    x = _apply_penalties(logits.clone(), row)
    if len(row.output) < p.min_tokens:
        for t in (*eos_ids, *p.stop_token_ids):
            x[t] = float("-inf")
    if p.temperature == 0:
        out = torch.zeros_like(x)
        out[int(x.argmax())] = 1.0
        return out
    return torch.softmax(_filter(x / p.temperature, p), dim=-1)


def accept(logits: torch.Tensor, drafts: list[int], q: torch.Tensor | None, candidates: torch.Tensor | None,
           row: RowState, eos_ids: tuple[int, ...]) -> tuple[int, list[int]]:
    """logits: FP32 [k + 1, vocab] for one sequence; drafts: k tokens; q, candidates: [k, top_k].
    Returns (accepted drafts, new tokens): the accepted drafts plus one token from the target."""
    k = len(drafts)
    if _plain(row):
        pred = logits.argmax(-1).tolist()
        n = 0
        while n < k and drafts[n] == pred[n]:
            n += 1
        return n, drafts[:n] + [pred[n]]
    gen = row.generator
    out: list[int] = []
    for i in range(k + 1):
        state = RowState(row.params, row.prompt, row.output + out, gen)
        p = _target_probs(logits[i], state, eos_ids)
        if i == k:
            out.append(int(torch.multinomial(p.cpu(), 1, generator=gen)))
            return k, out
        d = drafts[i]
        hit = (candidates[i] == d).nonzero()
        qd = float(q[i, hit[0, 0]]) if hit.numel() else 0.0
        u = float(torch.rand((), generator=gen))
        if qd > 0 and u * qd <= float(p[d]):
            out.append(d)
            continue
        qfull = torch.zeros_like(p)
        qfull.scatter_add_(0, candidates[i], q[i])
        residual = (p - qfull).clamp(min=0)
        total = float(residual.sum())
        out.append(int(torch.multinomial((residual / total if total > 0 else p).cpu(), 1, generator=gen)))
        return i, out
    return k, out


KMAX = 64


def device_accept(logits: torch.Tensor, drafts: torch.Tensor, q: torch.Tensor, cand: torch.Tensor,
                  temp: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor, min_p: torch.Tensor,
                  u_accept: torch.Tensor, u_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Acceptance for a batch of verify blocks on the device, without host synchronization (CUDA-graph safe).

    logits: FP32 [B, k + 1, V]; drafts: [B, k]; q, cand: [B, k, C] (draft probabilities over candidates);
    temp, top_p, min_p: [B]; top_k: int [B] in 1..KMAX (sampled rows); u_accept: [B, k], u_residual: [B, k + 1],
    uniform random numbers. Rows with temp == 0 are greedy. Returns (accepted drafts [B], new tokens [B, k + 1],
    of which the first accepted + 1 are valid).
    """
    b, k1, _ = logits.shape
    k = k1 - 1
    pred = logits.argmax(-1)
    greedy_match = (drafts == pred[:, :k]).long().cumprod(-1)
    n_greedy = greedy_match.sum(-1)

    kmax = min(KMAX, logits.shape[-1])
    vals, idx = torch.topk(logits, kmax, dim=-1)
    ranks = torch.arange(kmax, device=logits.device)
    safe_t = torch.where(temp > 0, temp, torch.ones_like(temp))
    vals = vals / safe_t[:, None, None]
    vals = vals.masked_fill(ranks >= top_k[:, None, None], float("-inf"))
    probs = torch.softmax(vals, dim=-1)
    keep = (torch.cumsum(probs, -1) - probs) < top_p[:, None, None]
    keep &= probs >= min_p[:, None, None] * probs[..., :1]
    keep[..., 0] = True
    probs = probs * keep
    probs = probs / probs.sum(-1, keepdim=True)
    pd = (probs[:, :k] * (idx[:, :k] == drafts[..., None])).sum(-1)
    qd = (q * (cand == drafts[..., None])).sum(-1)
    ok = (u_accept * qd <= pd) & (qd > 0)
    n_sampled = ok.long().cumprod(-1).sum(-1)
    q_on_idx = ((idx[:, :k, :, None] == cand[:, :, None, :]) * q[:, :, None, :]).sum(-1)
    q_on_idx = torch.cat((q_on_idx, torch.zeros_like(q_on_idx[:, :1])), dim=1)
    rows = torch.arange(b, device=logits.device)
    p_n = probs[rows, n_sampled]
    resid = (p_n - q_on_idx[rows, n_sampled]).clamp(min=0)
    resid = torch.where((n_sampled < k)[:, None] & (resid.sum(-1, keepdim=True) > 0), resid, p_n)
    cdf = torch.cumsum(resid / resid.sum(-1, keepdim=True), -1)
    pick = (cdf < u_residual[rows, n_sampled][:, None]).sum(-1).clamp(max=kmax - 1)
    sampled_next = idx[rows, n_sampled, pick]

    greedy = temp == 0
    n = torch.where(greedy, n_greedy, n_sampled)
    nxt = torch.where(greedy, pred[rows, n_greedy], sampled_next)
    new = torch.cat((drafts, torch.zeros_like(drafts[:, :1])), dim=1)
    new[rows, n] = nxt
    return n, new


STOP_SLOTS = 8


def cut_at_stops(n: torch.Tensor, new: torch.Tensor, stop: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Accepted drafts of each row after cutting its new tokens at the first stop token or at its token limit.

    n: [B] accepted drafts; new: [B, k + 1] (the first n + 1 valid); stop: [B, S] stop token ids (-1 pads);
    limit: [B] most tokens the row may still emit (at least 1). The tokens after a stop are dropped anyway; the cut
    makes the committed state end at the last emitted token, so a finished request stays usable by the prefix
    cache. CUDA-graph safe.
    """
    k1 = new.shape[1]
    idx = torch.arange(k1, device=new.device)
    hit = (new[..., None] == stop[:, None, :]).any(-1) & (idx <= n[:, None])
    first = torch.where(hit.any(-1), hit.int().argmax(-1), torch.full_like(n, k1))
    return torch.minimum(torch.minimum(n, first), limit - 1)


def cut_host(n: int, new: list[int], stop: set[int], limit: int) -> int:
    """cut_at_stops for one row on the host."""
    for j, t in enumerate(new[: n + 1]):
        if t in stop or j + 1 >= limit:
            return min(n, j)
    return n
