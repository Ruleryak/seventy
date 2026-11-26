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
    ranks: query heads split as evenly as possible (each rank keeps the KV heads its query heads use), and a split by
    KV head: with at least as many KV heads as ranks, the KV heads split as evenly as possible (each rank owns their
    whole groups); with fewer, each KV head goes to a set of ranks (split.balanced of the ranks), which share its
    query heads. The layout with fewer KV heads on the fullest rank wins, then the one with fewer owned query heads;
    with even splits both are the usual even split."""
    group = heads // kv

    def by_heads(r: int) -> HeadSplit:
        o0, o1 = balanced(heads, size, r)
        k0, k1 = o0 // group, -(-o1 // group)
        return HeadSplit((k0, k1), (k0 * group, k1 * group), (o0, o1))

    def by_kv(r: int) -> HeadSplit:
        k0, k1 = balanced(kv, size, r)
        return HeadSplit((k0, k1), (k0 * group, k1 * group), (k0 * group, k1 * group))

    def by_shared(r: int) -> HeadSplit:
        h = next(h for h in range(kv) if balanced(size, kv, h)[0] <= r < balanced(size, kv, h)[1])
        a, b = balanced(size, kv, h)
        o0, o1 = balanced(group, b - a, r - a)
        return HeadSplit((h, h + 1), (h * group, (h + 1) * group), (h * group + o0, h * group + o1))

    def cost(layout) -> tuple[int, int, int]:
        parts = [layout(r) for r in range(size)]
        return (max(p.num_kv for p in parts), max(p.num_own for p in parts), max(p.num_heads for p in parts))

    layouts = [by_heads] + ([by_kv] if kv >= size else [by_shared])
    return min(layouts, key=cost)(rank)


def max_head_split(heads: int, kv: int, size: int) -> tuple[int, int]:
    """(KV heads, computed query heads) of the fullest rank: what sizes the KV cache and the state of every rank."""
    parts = [head_split(heads, kv, r, size) for r in range(size)]
    return max(p.num_kv for p in parts), max(p.num_heads for p in parts)


def kv_groups(heads: int, kv: int, size: int) -> list[tuple[int, ...]] | None:
    """The groups of ranks that share a KV head and can split its tokens, in rank order (None: no split): with
    fewer KV heads than ranks, size a multiple of kv, and every rank keeping one KV head and computing all its query
    heads, with its own query heads the balanced part of the group's heads at its index in the group."""
    if kv >= size or size % kv:
        return None
    parts = [head_split(heads, kv, r, size) for r in range(size)]
    groups: dict[int, list[int]] = {}
    for r, p in enumerate(parts):
        if p.num_kv != 1 or p.num_heads != heads // kv:
            return None
        groups.setdefault(p.kv[0], []).append(r)
    out = [tuple(g) for _, g in sorted(groups.items())]
    if any(len(g) != size // kv for g in out):
        return None
    for g in out:
        for i, r in enumerate(g):
            if parts[r].local_own != balanced(heads // kv, len(g), i):
                return None
    return out


VOCAB_ALIGN = 64
MLP_ALIGN = 64
