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


@pytest.mark.parametrize("tokens,experts,k", [(1, 256, 8), (40, 256, 8), (64, 512, 10), (3, 16, 4)])
@pytest.mark.parametrize("renorm", [True, False])
def test_route_matches_softmax_topk(tokens, experts, k, renorm):
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(tokens + experts)
    logits = (torch.randn(tokens, experts, generator=g) * 2).cuda()
    ids, w = kernels.moe_route(logits, k, renorm)
    probs = torch.softmax(logits, dim=-1)
    want_w, want_ids = torch.topk(probs, k, dim=-1)
    if renorm:
        want_w = want_w / want_w.sum(-1, keepdim=True)
    assert torch.equal(ids.long(), want_ids)
    torch.testing.assert_close(w, want_w, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_route_reads_strided_logits_and_breaks_ties_low(tokens):
    """Logits that are the first columns of a wider product (row stride > experts), with exact ties."""
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(tokens)
    wide = (torch.randint(-3, 4, (tokens, 513), generator=g).float()).cuda()
    ids, w = kernels.moe_route(wide, 10, True, experts=512)
    logits = wide[:, :512].cpu()
    for t in range(tokens):
        order = sorted(range(512), key=lambda e: (-logits[t, e].item(), e))[:10]
        assert ids[t].tolist() == order
        picked = logits[t, order]
        torch.testing.assert_close(w[t].cpu(), torch.softmax(picked, -1), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shared", [True, False])
def test_finish_adds_local_picks_and_the_gated_shared_expert(shared):
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(int(shared))
    t, k, dim, e0, e1 = 6, 10, 256, 128, 256
    ids = torch.randint(0, 512, (t * k,), generator=g, dtype=torch.int32)
    y = torch.randn(t * k, dim, generator=g)
    y[(ids < e0) | (ids >= e1)] = float("nan")
    sh = torch.randn(t, dim, generator=g).half()
    gate = torch.randn(t, 3, generator=g)
    out = kernels.moe_finish(y.cuda(), ids.cuda(), k, e0, e1, sh.cuda() if shared else None,
                             gate.cuda()[:, 1] if shared else None)
    local = ((ids >= e0) & (ids < e1)).view(t, k, 1)
    want = torch.where(local, y.view(t, k, dim), torch.zeros(())).sum(1)
    if shared:
        want = want + sh.float() * torch.sigmoid(gate[:, 1:2])
    torch.testing.assert_close(out.float().cpu(), want.half().float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("fmt", ["fp16", "nvfp4", "nvfp4-offload"])
def test_gpu_moe_matches_the_per_expert_loop(fmt):
    """MoE._small (one product for the router and the shared gate, warp route, moe_finish) and MoE._large (the
    prefill path: compact rows, the PyTorch plan) against the per-expert loop (MoE._loop) on random experts of a
    small config, FP16 (seventy_moe_gemm) or NVFP4 repacked for the tensor cores (seventy_moe_mma; offload: 6 of
    the 16 experts in pinned host memory, read over PCIe), with BF16 router weights that need the power-of-two
    scale."""
    from types import SimpleNamespace

    from seventy.model.moe import ExpertStack, MoE

    g = torch.Generator().manual_seed(3)
    hidden, width, experts, local = 256, 64, 32, 16
    m = MoE.__new__(MoE)
    m.cfg = SimpleNamespace(num_experts=experts, experts_per_token=4, norm_topk_prob=True)
    m.top_k, m.e0, m.e1, m.dtype, m.device = 4, 8, 24, torch.float16, torch.device("cuda")
    router = (torch.randn(experts, hidden, generator=g) * 0.05).bfloat16().float()
    router[0, :8] = torch.tensor([5e-8, -7e-8, 3e-7, 2.375, -1.5, 1e-6, 0.0, 4e-5]).bfloat16().float()
    m.router = router.cuda()
    m.shared_gate = (torch.randn(1, hidden, generator=g) * 0.05).bfloat16().float().cuda()
    m.gates, m.gate_scale = None, 1.0
    want_logits = torch.cat((m.router, m.shared_gate))
    assert m.use_fp16_gates() and m.gate_scale < 1.0
    assert torch.equal(torch.cat((m.router, m.shared_gate)).float() * m.gate_scale, want_logits)
    if fmt == "fp16":
        m.gate_up = ExpertStack(weight=(torch.randn(local, 2 * width, hidden, generator=g) * 0.05).half().cuda())
        m.down = ExpertStack(weight=(torch.randn(local, hidden, width, generator=g) * 0.05).half().cuda())
    else:
        def nv(n, k):
            return ExpertStack(packed=torch.randint(0, 256, (local, n, k // 2), generator=g, dtype=torch.uint8).cuda(),
                               scales=torch.randint(0x30, 0x40, (local, n, k // 16), generator=g,
                                                    dtype=torch.uint8).cuda(),
                               alpha=(torch.rand(local, n, generator=g) * 0.002 + 0.0005).cuda())

        m.gate_up, m.down = nv(2 * width, hidden), nv(hidden, width)
        on_gpu = 10 if fmt == "nvfp4-offload" else None
        m.gate_up.to_gemm(on_gpu)
        m.down.to_gemm(on_gpu)
        if on_gpu:
            assert m.gate_up.host[0].is_pinned() and m.gate_up.gemm_ptrs is not None
    wgu = (torch.randn(2 * width, hidden, generator=g) * 0.05).half().cuda()
    wd = (torch.randn(hidden, width, generator=g) * 0.05).half().cuda()
    m.shared_gate_up = lambda x: x @ wgu.T
    m.shared_down = lambda x: x @ wd.T
    for tokens in (1, 7, 64, 300):
        x = torch.randn(tokens, hidden, generator=g).half().cuda()
        got = m(x)
        ids, weights = m.route(x)
        want = m._loop(x, ids, weights, m._shared(x)).to(torch.float16)
        torch.testing.assert_close(got.float(), want.float(), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("pairs,per", [(30, 8), (3000, 8), (40960, 8)])
def test_large_plan_slots_cover_each_local_pair_once(pairs, per):
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(pairs)
    ids = torch.randint(0, 512, (pairs,), generator=g, dtype=torch.int32).cuda()
    e0, e1 = 128, 256
    (order, se, ss, sc, num, max_slots), rows, local = kernels.moe_plan_large(ids, e0, e1, per)
    n = int(num.item())
    assert n <= max_slots and int(local) == int(((ids >= e0) & (ids < e1)).sum())
    seen = []
    for s in range(n):
        c = int(sc[s])
        assert 1 <= c <= per
        for r in range(int(ss[s]), int(ss[s]) + c):
            p = int(order[r])
            assert e0 + int(se[s]) == int(ids[p]) and int(rows[p]) == r
            seen.append(p)
    assert sorted(seen) == sorted(((ids >= e0) & (ids < e1)).nonzero().view(-1).tolist())
