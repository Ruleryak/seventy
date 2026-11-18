"""The input of one model step: a flat batch of new tokens from one or more sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SequenceChunk:
    """The new tokens of one sequence in a step.

    context_len tokens of the sequence are already in the cache. block_table lists the cache blocks of the
    sequence in order; it must cover context_len + len(token_ids) tokens. state_slot is the sequence's slot in the
    Gated DeltaNet state cache. positions ([3][n], temporal/height/width) default to consecutive text positions.
    logits is "last", "all" or "none".
    """

    token_ids: list[int]
    context_len: int
    block_table: list[int]
    state_slot: int
    positions: list[list[int]] | None = None
    logits: str = "last"


@dataclass
class StepInput:
    """Tensors and host lists for one step.

    Sequence b owns token rows [query_start[b], query_start[b + 1]). logit_rows are the token rows whose logits
    the step returns, in order.
    """

    token_ids: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor
    logit_rows: torch.Tensor
    query_start: list[int]
    context_lens: list[int]
    state_slots: list[int]
    block_size: int
    embeds: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return self.query_start[-1]

    @property
    def num_seqs(self) -> int:
        return len(self.context_lens)

    @staticmethod
    def build(chunks: list[SequenceChunk], block_size: int, device: torch.device | str) -> "StepInput":
        ids, pos, starts, rows = [], [[], [], []], [0], []
        width = max((len(c.block_table) for c in chunks), default=0)
        tables = torch.zeros(len(chunks), max(width, 1), dtype=torch.int32)
        for b, c in enumerate(chunks):
            n = len(c.token_ids)
            need = -(-(c.context_len + n) // block_size)
            if len(c.block_table) < need:
                raise ValueError(f"sequence {b} needs {need} cache blocks, has {len(c.block_table)}")
            ids += c.token_ids
            p = c.positions or [list(range(c.context_len, c.context_len + n))] * 3
            for axis in range(3):
                pos[axis] += p[axis]
            if c.logits == "all":
                rows += range(starts[-1], starts[-1] + n)
            elif c.logits == "last" and n:
                rows.append(starts[-1] + n - 1)
            starts.append(starts[-1] + n)
            tables[b, : len(c.block_table)] = torch.tensor(c.block_table, dtype=torch.int32)
        return StepInput(
            token_ids=torch.tensor(ids, dtype=torch.long, device=device),
            positions=torch.tensor(pos, dtype=torch.long, device=device),
            block_tables=tables.to(device),
            logit_rows=torch.tensor(rows, dtype=torch.long, device=device),
            query_start=starts,
            context_lens=[c.context_len for c in chunks],
            state_slots=[c.state_slot for c in chunks],
            block_size=block_size,
        )
