"""Linear layers of FP8 checkpoint modules (one scale per tensor, as in ModelOpt FP8) stay FP8 for the FP8 GEMM, with
a dense FP16 tail for dense modules after them, and give the products of the dequantized weights."""

import pytest
import torch
from safetensors.torch import save_file

from seventy.checkpoint import Checkpoint
from seventy.model.layers import Linear, Source
from seventy.ops import kernels_available

pytestmark = pytest.mark.gpu
if not (torch.cuda.is_available() and kernels_available()):
    pytestmark = [pytestmark, pytest.mark.skip(reason="needs a CUDA GPU and the kernel library")]


def test_fp8_modules_with_a_dense_tail(tmp_path):
    g = torch.Generator().manual_seed(2)
    a = torch.randn(96, 128, generator=g) / 0.07
    b = torch.randn(64, 128, generator=g) / 0.07
    save_file({"a.weight": a.to(torch.float8_e4m3fn), "a.weight_scale": torch.tensor(0.01),
               "b.weight": b.to(torch.float8_e4m3fn), "b.weight_scale": torch.tensor(0.02),
               "c.weight": torch.randn(8, 128, generator=g).to(torch.bfloat16)}, str(tmp_path / "model.safetensors"))
    with Checkpoint(tmp_path) as ckpt:
        lin = Linear(ckpt, [Source("a", ((16, 80),)), Source("b"), Source("c")], torch.float16, "cuda")
        assert lin.fp8 is not None and lin.fp8_tail is not None and lin.out_features == 64 + 64 + 8
        want = torch.cat((ckpt.linear_part("a", rows=(16, 80)), ckpt.linear_part("b"), ckpt.linear_part("c").float()))
    for rows in (1, 7, 64, 100):
        x = torch.randn(rows, 128, generator=g).half().cuda()
        ref = x.float() @ want.cuda().T
        got = lin(x).float()
        assert (got - ref).abs().max().item() < 2e-3 * ref.abs().max().item() + 1e-2


def test_mxfp8_module_with_one_scale_per_row(tmp_path):
    g = torch.Generator().manual_seed(3)
    w = torch.randn(64, 128, generator=g)
    exps = torch.randint(122, 128, (64, 4), generator=g, dtype=torch.int32).to(torch.uint8)
    values = (w * 8).to(torch.float8_e4m3fn)
    save_file({"m.weight": values, "m.weight_scale": exps}, str(tmp_path / "model.safetensors"))
    with Checkpoint(tmp_path) as ckpt:
        exact = ckpt.linear_part("m").float()
        lin = Linear(ckpt, [Source("m")], torch.float16, "cuda")
        assert lin.fp8 is not None
        rows_v, rows_s = ckpt.mxfp8_rows("m")
    approx = rows_v.view(torch.float8_e4m3fn).float() * rows_s[:, None]
    assert ((approx - exact).norm() / exact.norm()).item() < 1e-2
    x = torch.randn(3, 128, generator=g).half().cuda()
    ref = x.float() @ exact.cuda().T
    assert (lin(x).float() - ref).abs().max().item() < 2e-2 * ref.abs().max().item()
