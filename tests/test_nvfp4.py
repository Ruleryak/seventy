import pytest
import torch

from seventy import nvfp4
from seventy.nvfp4 import Nvfp4Weight


def test_unpack_nibble_order():
    packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)
    assert nvfp4.unpack(packed).tolist() == [[0.5, 1.0, -0.0, -6.0]]


def test_pack_roundtrip():
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(16, 16)
    assert torch.equal(nvfp4.pack(nvfp4.unpack(codes)), codes)


def test_dequantize_matches_definition():
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (4, 16), dtype=torch.uint8)
    scales = torch.tensor([0x38, 0x40, 0x01, 0x7E] * 2, dtype=torch.uint8).reshape(4, 2)
    w = Nvfp4Weight(packed, scales, 3.0)
    dense = nvfp4.dequantize(w, torch.float32)
    values = nvfp4.unpack(packed)
    expected = torch.empty(4, 32)
    for i in range(4):
        for j in range(32):
            s = scales[i, j // 16].view(torch.float8_e4m3fn).float().item()
            expected[i, j] = values[i, j] * (torch.tensor(s) / torch.tensor(3.0))
    assert torch.equal(dense, expected)
    assert scales[0, 0].view(torch.float8_e4m3fn).float().item() == 1.0
    assert scales[1, 1].view(torch.float8_e4m3fn).float().item() == 448.0


def test_quantize_error_is_small():
    torch.manual_seed(1)
    w = torch.randn(64, 256) * 0.02
    q = nvfp4.quantize(w)
    err = (nvfp4.dequantize(q, torch.float32) - w).norm() / w.norm()
    assert err < 0.12


def test_slices():
    torch.manual_seed(2)
    q = nvfp4.quantize(torch.randn(8, 64))
    full = nvfp4.dequantize(q, torch.float32)
    assert torch.equal(nvfp4.dequantize(q.rows(2, 5), torch.float32), full[2:5])
    assert torch.equal(nvfp4.dequantize(q.cols(16, 48), torch.float32), full[:, 16:48])
    with pytest.raises(ValueError):
        q.cols(8, 24)


def test_shape_checks():
    with pytest.raises(ValueError):
        Nvfp4Weight(torch.zeros(2, 8, dtype=torch.uint8), torch.zeros(2, 2, dtype=torch.uint8), 1.0)
