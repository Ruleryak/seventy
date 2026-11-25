"""The mixture-of-experts kernels (csrc/moe.cu): the plan sorts the pairs of the local experts by expert, in a
stable order, into slots; the grouped GEMM gives each pair its expert's product, for NVFP4 and FP16 weights, the
same as a PyTorch product of the dequantized weights."""

import pytest
import torch

from seventy import nvfp4
from seventy.ops import kernels_available

pytestmark = pytest.mark.gpu
if not (torch.cuda.is_available() and kernels_available()):
    pytestmark = [pytestmark, pytest.mark.skip(reason="needs a CUDA GPU and the kernel library")]


def routes(tokens, k, experts, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(experts, generator=g)[:k] for _ in range(tokens)]).int().cuda()


@pytest.mark.parametrize("tokens,k,experts,e0,e1", [(1, 8, 256, 0, 256), (40, 8, 256, 64, 128), (64, 10, 512, 0, 171),
                                                    (3, 8, 16, 5, 9)])
def test_plan_sorts_local_pairs_into_slots(tokens, k, experts, e0, e1):
    from seventy.ops import kernels

    ids = routes(tokens, k, experts, tokens).reshape(-1)
    order, slot_expert, slot_start, slot_count, num, max_slots = kernels.moe_plan(ids, e0, e1)
    flat = ids.cpu().tolist()
    local = sorted((e - e0, p) for p, e in enumerate(flat) if e0 <= e < e1)
    n = len(local)
    assert order[:n].cpu().tolist() == [p for _, p in local]
    s = int(num.item())
    seen = []
    for i in range(s):
        e, a, c = int(slot_expert[i]), int(slot_start[i]), int(slot_count[i])
        assert 1 <= c <= kernels.MOE_ROWS_PER_SLOT
        assert all(flat[p] - e0 == e for p in order[a : a + c].cpu().tolist())
        seen += order[a : a + c].cpu().tolist()
    assert sorted(seen) == sorted(p for _, p in local) and s <= max_slots


@pytest.mark.parametrize("fmt", ["nvfp4", "fp16"])
@pytest.mark.parametrize("tokens,k,n,kk", [(1, 8, 1024, 2048), (37, 8, 2048, 512), (64, 10, 1280, 2560)])
def test_grouped_gemm_matches_dense_products(fmt, tokens, k, n, kk):
    from seventy.ops import kernels

    experts, e0, e1 = 16, 4, 12
    g = torch.Generator().manual_seed(3)
    ids = routes(tokens, k, experts, 7)
    x = (torch.randn(tokens, kk, generator=g) * 0.5).half().cuda()
    dense = []
    if fmt == "nvfp4":
        parts = [nvfp4.quantize(torch.randn(n, kk, generator=g)) for _ in range(e1 - e0)]
        packed = torch.stack([p.packed for p in parts]).cuda()
        scales = torch.stack([p.scales for p in parts]).cuda()
        alpha = torch.stack([torch.full((n,), 1.0 / p.global_scale) for p in parts]).float().cuda()
        dense = [nvfp4.dequantize(p, torch.float32).cuda() for p in parts]
        weight = packed
    else:
        weight = (torch.randn(e1 - e0, n, kk, generator=g) * 0.05).half().cuda()
        scales = alpha = None
        dense = [weight[e].float() for e in range(e1 - e0)]
    plan = kernels.moe_plan(ids.reshape(-1), e0, e1)
    pw = torch.rand(tokens * k, generator=g).cuda()
    got16 = kernels.moe_gemm(x, k, weight, scales, alpha, plan, tokens * k)
    got32 = kernels.moe_gemm(x, k, weight, scales, alpha, plan, tokens * k, pair_weight=pw)
    flat = ids.reshape(-1).cpu().tolist()
    for p, e in enumerate(flat):
        if not e0 <= e < e1:
            assert not got16[p].any() and not got32[p].any()
            continue
        ref = x[p // k].float() @ dense[e - e0].T
        tol = 2e-3 * ref.abs().max().item() + 1e-3
        assert (got16[p].float() - ref).abs().max().item() < tol + 2e-3 * ref.abs().max().item()
        assert (got32[p] - ref * pw[p]).abs().max().item() < tol
