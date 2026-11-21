"""seventy_topk (csrc/topk.cu) against torch: values, and ids by value descending then id ascending."""

import pytest
import torch

from seventy.ops import kernels

pytestmark = pytest.mark.gpu


def reference(x, k, id_offset=0, ids=None):
    rows, n = x.shape
    if ids is None:
        ids = torch.arange(n, dtype=torch.int64).expand(rows, n) + id_offset
    out_v, out_i = [], []
    for r in range(rows):
        keep = ids[r] >= 0
        v, i = x[r][keep].tolist(), ids[r][keep].tolist()
        order = sorted(range(len(v)), key=lambda j: (-v[j], i[j]))[:k]
        vv = [v[j] for j in order] + [float("-inf")] * (k - len(order))
        ii = [i[j] for j in order] + [-1] * (k - len(order))
        out_v.append(vv)
        out_i.append(ii)
    return torch.tensor(out_v), torch.tensor(out_i)


@pytest.mark.parametrize("rows, n, k", [(1, 10, 4), (3, 64, 64), (5, 1000, 16), (32, 62080, 64), (28, 62080, 16),
                                        (2, 8192, 64), (2, 8193, 7), (4, 30, 64), (2, 100000, 33)])
@pytest.mark.parametrize("kind", ["randn", "ties", "equal", "neg"])
def test_topk_matches_reference(rows, n, k, kind):
    g = torch.Generator().manual_seed(rows * 1000 + n + k)
    if kind == "randn":
        x = torch.randn(rows, n, generator=g) * 5
    elif kind == "ties":
        x = torch.randint(-20, 20, (rows, n), generator=g).float()
    elif kind == "equal":
        x = torch.full((rows, n), 1.5)
    else:
        x = -torch.rand(rows, n, generator=g) - 1e-3
    v, i = kernels.topk(x.cuda(), k, id_offset=123)
    rv, ri = reference(x, k, 123)
    assert torch.equal(v.cpu(), rv.float())
    assert torch.equal(i.cpu().long(), ri)


def test_topk_with_ids_and_padding():
    g = torch.Generator().manual_seed(7)
    x = torch.randint(-5, 5, (6, 256), generator=g).float()
    ids = torch.randperm(10000, generator=g)[: 6 * 256].view(6, 256).int()
    ids[:, ::7] = -1
    v, i = kernels.topk(x.cuda(), 64, ids=ids.cuda())
    rv, ri = reference(x, 64, ids=ids.long())
    assert torch.equal(v.cpu(), rv.float())
    assert torch.equal(i.cpu().long(), ri)
