"""Sampling parameters and the sampler: from FP32 logits to the next token of each sequence."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    """How one request generates tokens. temperature 0 means greedy. top_k 0 and top_p 1 turn those filters off."""

    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens: int = 0

    def check(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        for n in (self.logprobs, self.prompt_logprobs):
            if n is not None and not 0 <= n <= 20:
                raise ValueError("logprobs and prompt_logprobs must be between 0 and 20")


@dataclass
class TokenLogprobs:
    """Log probability of the chosen token and the top-n alternatives (raw model distribution)."""

    token: int
    logprob: float
    rank: int
    top: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class RowState:
    """What the sampler needs to know about one row: its parameters, earlier tokens, and its random generator."""

    params: SamplingParams
    prompt: list[int]
    output: list[int]
    generator: torch.Generator | None


def _apply_penalties(logits: torch.Tensor, row: RowState) -> torch.Tensor:
    p = row.params
    if p.repetition_penalty != 1.0:
        seen = torch.tensor(sorted(set(row.prompt) | set(row.output)), dtype=torch.long, device=logits.device)
        if seen.numel():
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
    if (p.presence_penalty or p.frequency_penalty) and row.output:
        ids, counts = torch.tensor(row.output, device=logits.device).unique(return_counts=True)
        logits[ids] -= p.presence_penalty + p.frequency_penalty * counts.to(logits.dtype)
    return logits


def _filter(logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    if p.top_k and p.top_k < logits.numel():
        kth = torch.topk(logits, p.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if p.top_p < 1.0 or p.min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        if p.min_p > 0.0:
            logits = logits.masked_fill(probs < p.min_p * probs.max(), float("-inf"))
        if p.top_p < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            drop = (cum - sorted_probs) > p.top_p
            logits = logits.masked_fill(torch.zeros_like(drop).scatter(0, order, drop), float("-inf"))
    return logits


def logprobs_of(logits: torch.Tensor, tokens: torch.Tensor, n: int) -> list[TokenLogprobs]:
    """Raw log probabilities of `tokens` (one per row) and the top-n of each row."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    chosen = lp.gather(1, tokens[:, None])[:, 0]
    rank = (lp > chosen[:, None]).sum(-1) + 1
    top_v, top_i = torch.topk(lp, max(n, 1), dim=-1)
    out = []
    for r in range(logits.shape[0]):
        top = list(zip(top_i[r, :n].tolist(), top_v[r, :n].tolist())) if n else []
        out.append(TokenLogprobs(int(tokens[r]), float(chosen[r]), int(rank[r]), top))
    return out


def sample(logits: torch.Tensor, rows: list[RowState], eos_ids: tuple[int, ...] = ()) -> tuple[list[int], list]:
    """The next token of each row, and its logprobs (None for rows that did not ask for them)."""
    tokens = []
    greedy = all(r.params.temperature == 0 and r.params.repetition_penalty == 1.0 and not r.params.presence_penalty
                 and not r.params.frequency_penalty and len(r.output) >= r.params.min_tokens for r in rows)
    if greedy:
        chosen = logits.argmax(-1)
        tokens = chosen.tolist()
    else:
        for i, row in enumerate(rows):
            p = row.params
            x = logits[i].clone()
            x = _apply_penalties(x, row)
            if len(row.output) < p.min_tokens:
                for t in (*eos_ids, *p.stop_token_ids):
                    x[t] = float("-inf")
            if p.temperature == 0:
                tokens.append(int(x.argmax()))
                continue
            x = _filter(x / p.temperature, p)
            probs = torch.softmax(x, dim=-1)
            tokens.append(int(torch.multinomial(probs, 1, generator=row.generator)))
        chosen = torch.tensor(tokens, device=logits.device)
    lps = [None] * len(rows)
    want = [i for i, r in enumerate(rows) if r.params.logprobs is not None]
    if want:
        n = max(rows[i].params.logprobs for i in want)
        got = logprobs_of(logits[want], chosen[want], n)
        for i, lp in zip(want, got):
            lp.top = lp.top[: rows[i].params.logprobs]
            lps[i] = lp
    return tokens, lps
