"""Weight formats of the checkpoint reader: CSF-coded NVFP4 block scales (seventy.checkpoint.csf_decode), ModelOpt
NVFP4, FP8 and MXFP8 modules in a small safetensors file, against their definitions."""

import torch
from safetensors.torch import save_file

from seventy import nvfp4
from seventy.checkpoint import Checkpoint, csf_decode


def csf_encode(scales: torch.Tensor, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The CSF code of uint8 scales [rows, cols]: per block of rows, each row's minimum as its base, 4-bit offsets
    (low nibble first), and exceptions (value << 24 | flat index) for offsets above 15."""
    rows, cols = scales.shape
    per = rows // blocks
    s = scales.to(torch.int32).view(blocks, per, cols)
    base = s.amin(dim=-1)
    off = s - base[..., None]
    big = off > 15
    flat = torch.nonzero(big.reshape(-1)).reshape(-1)
    exceptions = ((scales.reshape(-1)[flat].to(torch.int64) << 24) | flat).to(torch.int64)
    off = off.masked_fill(big, 0).to(torch.uint8).view(blocks, per * cols)
    body = off[:, 0::2] | (off[:, 1::2] << 4)
    fixed = torch.cat((base.to(torch.uint8), body), dim=1)
    return fixed, exceptions.to(torch.uint32 if hasattr(torch, "uint32") else torch.int64).view(torch.int32) \
        if exceptions.numel() else torch.zeros(0, dtype=torch.int32)


def test_csf_decode_round_trip():
    g = torch.Generator().manual_seed(0)
    scales = torch.randint(90, 112, (64, 40), generator=g, dtype=torch.int32)
    scales[3, 7] = 140
    scales[60, 0] = 20
    scales = scales.to(torch.uint8)
    fixed, exceptions = csf_encode(scales, 4)
    assert fixed.shape == (4, 16 + 16 * 40 // 2)
    assert torch.equal(csf_decode(fixed, exceptions, 64, 40), scales)


def test_modelopt_fp8_and_mxfp8_modules(tmp_path):
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 128, generator=g)
    fp8 = (w / 0.05).to(torch.float8_e4m3fn)
    exps = torch.randint(120, 130, (64, 4), generator=g, dtype=torch.int32).to(torch.uint8)
    mx = (w * 8).to(torch.float8_e4m3fn)
    q = nvfp4.quantize(w)
    save_file({
        "a.weight": fp8, "a.weight_scale": torch.tensor(0.05), "a.input_scale": torch.tensor(1.0),
        "b.weight": mx, "b.weight_scale": exps,
        "c.weight": q.packed, "c.weight_scale": q.scales.view(torch.float8_e4m3fn),
        "c.weight_scale_2": torch.tensor(1.0 / q.global_scale), "c.input_scale": torch.tensor(1.0),
        "d.weight": w.to(torch.bfloat16),
    }, str(tmp_path / "model.safetensors"))
    with Checkpoint(tmp_path) as ckpt:
        assert [ckpt.kind(m) for m in "abcd"] == ["fp8", "mxfp8", "modelopt-nvfp4", "dense"]
        torch.testing.assert_close(ckpt.linear_part("a"), fp8.float() * 0.05)
        expect = (mx.float().view(64, 4, 32) * torch.exp2(exps.float() - 127)[..., None]).view(64, 128)
        torch.testing.assert_close(ckpt.linear_part("b", rows=(8, 24), cols=(32, 96)), expect[8:24, 32:96])
        got = ckpt.nvfp4_part("c", rows=(16, 48), cols=(32, 128))
        torch.testing.assert_close(nvfp4.dequantize(got, torch.float32),
                                   nvfp4.dequantize(q, torch.float32)[16:48, 32:128], rtol=1e-6, atol=1e-6)
        assert ckpt.out_rows("c") == 64 and ckpt.is_nvfp4("c") and not ckpt.is_nvfp4("a")
