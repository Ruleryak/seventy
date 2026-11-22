"""seventy.upload.to_device: the same bytes on the GPU as Tensor.to, for several dtypes, tensors of several chunks
(both pinned buffers in turn), small tensors, and tensors that are not contiguous."""

import pytest
import torch

from seventy import upload

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.uint8, torch.int32])
def test_same_bytes_as_tensor_to(dtype):
    g = torch.Generator().manual_seed(0)
    for shape in ((1000,), (3000, 1001), (upload.CHUNK_BYTES // 2 + 12345,)):
        if dtype.is_floating_point:
            t = torch.randn(shape, generator=g).to(dtype)
        else:
            t = torch.randint(0, 120, shape, generator=g).to(dtype)
        got = upload.to_device(t, "cuda:0")
        assert got.device == torch.device("cuda", 0) and got.dtype == dtype and got.shape == t.shape
        assert got.is_contiguous()
        assert torch.equal(got.cpu().view(torch.uint8), t.contiguous().view(torch.uint8))
    upload.release()


def test_not_contiguous_and_rows():
    t = torch.arange(4096 * 300, dtype=torch.float32).reshape(4096, 300)
    for part in (t.t(), t[:, 10:200], t[100:3000]):
        got = upload.to_device(part, "cuda:0")
        assert got.is_contiguous() and torch.equal(got.cpu(), part)
    upload.release()


def test_small_and_cpu_targets_use_tensor_to():
    t = torch.ones(16)
    assert torch.equal(upload.to_device(t, "cpu"), t)
    assert upload.to_device(t, "cuda:0").device.type == "cuda"
    assert upload._stage is None
