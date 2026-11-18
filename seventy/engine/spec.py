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
            out.append(int(torch.multinomial(p, 1, generator=gen)))
            return k, out
        d = drafts[i]
        hit = (candidates[i] == d).nonzero()
        qd = float(q[i, hit[0, 0]]) if hit.numel() else 0.0
        u = float(torch.rand((), device=logits.device, generator=gen))
        if qd > 0 and u * qd <= float(p[d]):
            out.append(d)
            continue
        qfull = torch.zeros_like(p)
        qfull.scatter_add_(0, candidates[i], q[i])
        residual = (p - qfull).clamp(min=0)
        total = float(residual.sum())
        out.append(int(torch.multinomial(residual / total if total > 0 else p, 1, generator=gen)))
        return i, out
    return k, out
