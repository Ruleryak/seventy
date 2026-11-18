import pytest
import torch

from seventy import nvfp4
from seventy.ops import kernels

pytestmark = pytest.mark.gpu


def make_parts(rows, k, seed):
    torch.manual_seed(seed)
    return [nvfp4.quantize(torch.randn(r, k) * (0.02 * (i + 1))) for i, r in enumerate(rows)]


def reference(x, parts):
    dense = torch.cat([nvfp4.dequantize(p, torch.float32) for p in parts])
    return x.float().cpu() @ dense.T


@pytest.mark.parametrize("m", [1, 5, 8, 13, 32, 64])
@pytest.mark.parametrize("rows, k", [([96], 64), ([40, 30], 1536), ([12, 12], 5120)])
def test_gemm_matches_reference(m, rows, k):
    parts = make_parts(rows, k, seed=m * 7 + k)
    w = nvfp4.pack_for_gemm([p.to("cuda") for p in parts])
    x = (torch.randn(m, k) * 2).half()
    out = kernels.nvfp4_gemm(x.cuda(), w).float().cpu()
    ref = reference(x, parts)
    err = (out - ref).abs().max().item()
    assert err <= 2e-3 * ref.abs().max().item() + 1e-3, err


@pytest.mark.parametrize("splits", [1, 3, 80])
def test_splits_agree(splits):
    parts = make_parts([200], 1024, seed=11)
    w = nvfp4.pack_for_gemm([parts[0].to("cuda")])
    x = torch.randn(9, 1024).half().cuda()
    base = kernels.nvfp4_gemm(x, w, splits=1)
    torch.testing.assert_close(kernels.nvfp4_gemm(x, w, splits=splits), base, rtol=2e-3, atol=2e-3)


def test_model_shape_from_checkpoint(model_path):
    from seventy.checkpoint import Checkpoint

    with Checkpoint(model_path) as ckpt:
        p = "model.language_model.layers.0.mlp"
        parts = [ckpt.nvfp4_part(f"{p}.gate_proj", rows=(0, 4352)), ckpt.nvfp4_part(f"{p}.up_proj", rows=(0, 4352))]
    w = nvfp4.pack_for_gemm([q.to("cuda") for q in parts])
    torch.manual_seed(0)
    x = torch.randn(16, 5120).half()
    out = kernels.nvfp4_gemm(x.cuda(), w).float().cpu()
    ref = reference(x, parts)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 1e-3, rel


def test_dequantize_from_gemm_layout():
    parts = make_parts([40, 30], 1536, seed=5)
    w = nvfp4.pack_for_gemm([p.to("cuda") for p in parts], block_rows=32)
    dense = kernels.nvfp4_gemm_dequantize(w).cpu()
    ref = torch.cat([nvfp4.dequantize(p, torch.float32) for p in parts]).half()
    diff = (dense.float() - ref.float()).abs()
    assert (diff <= ref.float().abs() * 1e-3 + 1e-7).all()


def test_pack_blocks_agree():
    parts = make_parts([100, 60], 128, seed=6)
    a = nvfp4.pack_for_gemm([p.to("cuda") for p in parts], block_rows=32)
    b = nvfp4.pack_for_gemm([p.to("cuda") for p in parts], block_rows=4096)
    assert torch.equal(a.wpk, b.wpk) and torch.equal(a.wsc, b.wsc) and torch.equal(a.alpha, b.alpha)
