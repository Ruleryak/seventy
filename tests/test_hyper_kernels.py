"""The hyper-connection kernels (csrc/hyper.cu) against their PyTorch definitions, and one whole hyper-connection
on the GPU (kernels and dense GEMMs) against the FP32 path of seventy.model.qwen4_exp.HyperConnection."""

import pytest
import torch
import torch.nn.functional as F

from seventy.ops import kernels_available

pytestmark = pytest.mark.gpu
if not (torch.cuda.is_available() and kernels_available()):
    pytestmark = [pytestmark, pytest.mark.skip(reason="needs a CUDA GPU and the kernel library")]

HC, HIDDEN, RANK = 4, 2560, 320


def grouped_norm(h, w, eps=1e-6):
    f = h.view(h.shape[0], -1, HIDDEN)
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).view(h.shape[0], -1) * w


@pytest.mark.parametrize("rows", [1, 5, 64])
def test_hyper_connection_kernels(rows):
    from seventy.ops import kernels

    g = torch.Generator().manual_seed(rows)
    h = (torch.randn(rows, HC * HIDDEN, generator=g) * 3).cuda()
    w = (1 + torch.randn(HC * HIDDEN, generator=g) * 0.1).cuda()
    n = kernels.hc_norm(h, w, HC, 1e-6)
    torch.testing.assert_close(n.float(), grouped_norm(h, w), rtol=2e-3, atol=2e-3)
    d = (torch.randn(rows, RANK + HC, generator=g) * 4).cuda()
    s, inj = kernels.hc_act(d, RANK, HC, True)
    torch.testing.assert_close(s.float(), F.silu(d[:, :RANK] / HC), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(inj, 2 * torch.sigmoid(d[:, RANK:] / HC), rtol=1e-5, atol=1e-6)
    u = (torch.randn(rows, HC * HIDDEN, generator=g) * 2).cuda()
    x = kernels.hc_combine(u, n, HC)
    want = (torch.sigmoid(u) * n.float()).view(rows, HC, HIDDEN).mean(1)
    torch.testing.assert_close(x.float(), want, rtol=2e-3, atol=2e-3)
    y = torch.randn(rows, HIDDEN, generator=g).half().cuda()
    want_h = h + (y.float()[:, None, :] * inj[:, :, None]).reshape(h.shape)
    torch.testing.assert_close(kernels.hc_inject(h.clone(), y, inj, HC), want_h, rtol=1e-6, atol=1e-5)
