import pytest
import torch

from seventy import nvfp4
from seventy.nvfp4 import Nvfp4Weight
from seventy.ops import kernels, reference

pytestmark = pytest.mark.gpu


def fp16_close(a, b, ulps=1):
    diff = (a.float() - b.float()).abs()
    tol = ulps * torch.finfo(torch.float16).eps * b.float().abs().clamp(min=2**-14)
    return bool((diff <= tol + 1e-7).all()), float(diff.max())


@pytest.mark.parametrize("rows, dim", [(3, 5120), (7, 256), (5, 128), (4, 6), (1, 17408)])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rms_norm(rows, dim, with_residual):
    torch.manual_seed(rows * dim)
    dev = torch.device("cuda")
    x = (torch.randn(rows, dim) * 3).half()
    r = (torch.randn(rows, dim) * 3).half() if with_residual else None
    w = 1.0 + torch.randn(dim) * 0.1
    ref_out, ref_res = reference.rms_norm(x, w, 1e-6, r)
    out, res = kernels.rms_norm(x.to(dev), w.to(dev), 1e-6, None if r is None else r.to(dev))
    ok, worst = fp16_close(out.cpu(), ref_out, ulps=2)
    assert ok, worst
    if with_residual:
        assert torch.equal(res.cpu(), ref_res)


def test_rms_norm_in_place():
    torch.manual_seed(3)
    dev = torch.device("cuda")
    x = torch.randn(4, 512).half().to(dev)
    r = torch.randn(4, 512).half().to(dev)
    w = torch.ones(512, device=dev)
    ref_out, ref_res = reference.rms_norm(x.cpu(), w.cpu(), 1e-6, r.cpu())
    kernels.rms_norm(x, w, 1e-6, r, out=x, residual_out=r)
    assert fp16_close(x.cpu(), ref_out, ulps=2)[0]
    assert torch.equal(r.cpu(), ref_res)


def test_nvfp4_dequantize_bit_exact():
    torch.manual_seed(4)
    n, k = 64, 512
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    codes = codes[(codes != 0x7F) & (codes != 0xFF)]
    scales = codes[torch.randint(0, codes.numel(), (n, k // 16))]
    w = Nvfp4Weight(packed, scales, 1234.5)
    ref = reference.nvfp4_dequantize(w)
    out = kernels.nvfp4_dequantize(w.to("cuda"))
    assert torch.equal(out.cpu().view(torch.int16), ref.view(torch.int16))


def test_nvfp4_dequantize_quantized_weight():
    torch.manual_seed(5)
    q = nvfp4.quantize(torch.randn(48, 5120) * 0.02)
    assert torch.equal(kernels.nvfp4_dequantize(q.to("cuda")).cpu(), reference.nvfp4_dequantize(q))


def test_kernel_rejects_bad_arguments():
    w = torch.ones(6, device="cuda")
    with pytest.raises(ValueError):
        kernels.rms_norm(torch.ones(2, 6, device="cuda"), w, 1e-6)
    with pytest.raises(Exception, match="invalid argument"):
        kernels.rms_norm(torch.ones(2, 5, device="cuda", dtype=torch.float16), torch.ones(5, device="cuda"), 1e-6)


@pytest.mark.gpu
def test_silu_mul_matches_reference():
    from seventy import ops
    from seventy.ops import reference as R

    torch.manual_seed(4)
    x = (torch.randn(37, 2 * 4352) * 4).half()
    assert torch.equal(ops.silu_mul(x.cuda()).cpu(), R.silu_mul(x.cuda()).cpu())


@pytest.mark.gpu
@pytest.mark.parametrize("m", [1, 8, 28, 32, 64])
def test_dense_gemm_fp8_matches_its_dequantized_weights(m):
    torch.manual_seed(m)
    n, k = 600, 1536
    w = (torch.randn(n, k) * 0.02).half().cuda()
    w[3] *= 50
    x = torch.randn(m, k).half().cuda()
    q = kernels.DenseWeightFP8(w)
    ref = x.float() @ q.dequantize().T
    got = kernels.dense_gemm(x, q)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3 * ref.abs().max().item())
    fp16 = kernels.dense_gemm(x, kernels.DenseWeight(w))
    rel = ((got - fp16).norm() / fp16.norm()).item()
    assert rel < 0.05, rel
    again = kernels.DenseWeightFP8.from_dense(kernels.DenseWeight(w))
    assert torch.equal(again.wpk, q.wpk) and torch.equal(again.col_scale, q.col_scale)
    scaled = kernels.dense_gemm(x, q, row_scale=torch.full((m,), 2.0, device="cuda"))
    torch.testing.assert_close(scaled, 2 * got, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("m", [1, 9, 32, 57])
@pytest.mark.parametrize("warps, splits", [(2, 1), (4, 3), (8, 1), (8, 2), (2, 12)])
def test_dense_gemm_fp8_wk_splits_agree(m, warps, splits):
    """Every warp and split count gives the dequantized product; the per-tile counters are zero after each call, and
    a call gives the same bits each time."""
    torch.manual_seed(100 + m)
    n, k = 200, 1536
    w = (torch.randn(n, k) * 0.02).half().cuda()
    x = torch.randn(m, k).half().cuda()
    q = kernels.DenseWeightFP8(w)
    ref = x.float() @ q.dequantize().T
    got = kernels.dense_gemm(x, q, fp8_config=(warps, splits))
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3 * ref.abs().max().item())
    assert torch.equal(kernels.dense_gemm(x, q, fp8_config=(warps, splits)), got)
    half_out = kernels.dense_gemm(x, q, out_fp32=False, fp8_config=(warps, splits))
    assert torch.equal(half_out, got.half())
    assert not kernels._tile_counters(x.device, 1).any()
