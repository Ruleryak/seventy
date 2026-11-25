"""How tensor parallel divides counts over the ranks (seventy.split): even splits stay as they were, uneven parts
differ by at most one unit, and every query head (or Gated DeltaNet value head) is owned by exactly one rank, inside
whole groups of the KV heads (key heads) that its rank keeps."""

import pytest

from seventy.split import balanced, head_split, max_head_split

SHAPES = [(24, 4), (48, 16), (32, 8), (16, 2), (24, 2), (16, 4), (8, 8)]


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
def test_balanced_parts_cover_the_total(size):
    for total, align in ((17408, 64), (248320, 64), (4304, 8), (16, 1), (25600, 64)):
        parts = [balanced(total, size, r, align) for r in range(size)]
        assert parts[0][0] == 0 and parts[-1][1] == total
        assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
        widths = [b - a for a, b in parts]
        assert max(widths) - min(widths) <= align
        if total % size == 0 and (total // size) % align == 0:
            assert len(set(widths)) == 1
        else:
            assert all(w % align == 0 for w in widths)


def test_balanced_parts_of_the_27b_on_3_gpus():
    assert [balanced(17408, 3, r, 64) for r in range(3)] == [(0, 5824), (5824, 11648), (11648, 17408)]
    assert [balanced(248320, 3, r, 64) for r in range(3)] == [(0, 82816), (82816, 165568), (165568, 248320)]
    with pytest.raises(ValueError):
        balanced(100, 3, 0, 64)


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("heads,kv", SHAPES)
def test_every_head_is_owned_once_inside_whole_groups(heads, kv, size):
    if heads < size:
        pytest.skip("fewer heads than ranks")
    group = heads // kv
    owned = []
    for r in range(size):
        s = head_split(heads, kv, r, size)
        assert s.heads == (s.kv[0] * group, s.kv[1] * group)
        assert s.heads[0] <= s.own[0] < s.own[1] <= s.heads[1]
        assert s.local_own == (s.own[0] - s.heads[0], s.own[1] - s.heads[0])
        owned += range(*s.own)
    assert owned == list(range(heads))


@pytest.mark.parametrize("heads,kv", SHAPES)
def test_even_splits_are_unchanged(heads, kv):
    for size in (1, 2, 4, 8):
        if heads % size or kv % size:
            continue
        for r in range(size):
            s = head_split(heads, kv, r, size)
            part, kpart = heads // size, kv // size
            assert (s.own, s.heads, s.kv) == ((r * part, (r + 1) * part),) * 2 + ((r * kpart, (r + 1) * kpart),)
            assert not s.partial


def test_the_27b_and_its_drafter_on_3_gpus():
    attn = [head_split(24, 4, r, 3) for r in range(3)]
    assert [(s.kv, s.own) for s in attn] == [((0, 2), (0, 8)), ((1, 3), (8, 16)), ((2, 4), (16, 24))]
    gdn = [head_split(48, 16, r, 3) for r in range(3)]
    assert [(s.kv, s.heads, s.own) for s in gdn] == [((0, 6), (0, 18), (0, 16)), ((5, 11), (15, 33), (16, 32)),
                                                       ((10, 16), (30, 48), (32, 48))]
    drafter = [head_split(32, 8, r, 3) for r in range(3)]
    assert [(s.kv, s.own) for s in drafter] == [((0, 3), (0, 12)), ((3, 6), (12, 24)), ((6, 8), (24, 32))]
    assert not any(s.partial for s in drafter)
    assert max_head_split(24, 4, 3) == (2, 12) and max_head_split(48, 16, 3) == (6, 18)
