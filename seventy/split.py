"""How tensor parallel divides a model over the ranks (no PyTorch import: the control panel uses it too).

Counts that split evenly over the GPUs split evenly. Others (3 GPUs) split into parts of whole units whose sizes
differ by at most one unit, and grouped heads use HeadSplit."""

from __future__ import annotations

from dataclasses import dataclass


def balanced(total: int, parts: int, index: int, align: int = 1) -> tuple[int, int]:
    """Part index of total items split into parts: equal parts when total divides into parts that are multiples
    of align, else parts of whole align-sized units whose sizes differ by at most one unit (the first parts take
    the extra units)."""
    if total % parts == 0 and (total // parts) % align == 0:
        part = total // parts
        return index * part, (index + 1) * part
    if total % align:
        raise ValueError(f"{total} is not a multiple of {align}")
    units, extra = divmod(total // align, parts)
    start = index * units + min(index, extra)
    return start * align, (start + units + (index < extra)) * align


@dataclass(frozen=True)
class HeadSplit:
    """One rank's share of grouped heads (query heads that share KV heads, or Gated DeltaNet value heads that
    share key heads). The rank keeps the KV heads kv, computes the query heads heads (all query heads of its KV
    heads, so each KV head serves whole groups in the kernels), and adds the query heads own to the output. When
    the heads do not split evenly, own is a part of heads and neighbouring ranks keep some KV heads twice."""

    kv: tuple[int, int]
    heads: tuple[int, int]
    own: tuple[int, int]

    @property
    def num_kv(self) -> int:
        return self.kv[1] - self.kv[0]

    @property
    def num_heads(self) -> int:
        return self.heads[1] - self.heads[0]

    @property
    def num_own(self) -> int:
        return self.own[1] - self.own[0]

    @property
    def local_own(self) -> tuple[int, int]:
        """own as indices into this rank's computed heads."""
        return self.own[0] - self.heads[0], self.own[1] - self.heads[0]

    @property
    def partial(self) -> bool:
        return self.own != self.heads


def head_split(heads: int, kv: int, rank: int, size: int) -> HeadSplit:
    """The rank's HeadSplit of heads query heads in groups over kv KV heads. Two layouts are compared over all
    ranks: query heads split as evenly as possible (each rank keeps the KV heads its query heads use), and KV heads
    split as evenly as possible (each rank owns their whole groups). The layout with fewer KV heads on the fullest
    rank wins, then the one with fewer owned query heads; with even splits both are the usual even split."""
    group = heads // kv

    def by_heads(r: int) -> HeadSplit:
        o0, o1 = balanced(heads, size, r)
        k0, k1 = o0 // group, -(-o1 // group)
        return HeadSplit((k0, k1), (k0 * group, k1 * group), (o0, o1))

    def by_kv(r: int) -> HeadSplit:
        k0, k1 = balanced(kv, size, r)
        return HeadSplit((k0, k1), (k0 * group, k1 * group), (k0 * group, k1 * group))

    def cost(layout) -> tuple[int, int, int]:
        parts = [layout(r) for r in range(size)]
        return (max(p.num_kv for p in parts), max(p.num_own for p in parts), max(p.num_heads for p in parts))

    layouts = [by_heads] + ([by_kv] if kv >= size else [])
    return min(layouts, key=cost)(rank)


def max_head_split(heads: int, kv: int, size: int) -> tuple[int, int]:
    """(KV heads, computed query heads) of the fullest rank: what sizes the KV cache and the state of every rank."""
    parts = [head_split(heads, kv, r, size) for r in range(size)]
    return max(p.num_kv for p in parts), max(p.num_heads for p in parts)


VOCAB_ALIGN = 64
MLP_ALIGN = 64
