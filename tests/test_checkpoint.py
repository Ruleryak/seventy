import pytest
import torch

from seventy import nvfp4
from seventy.checkpoint import Checkpoint

pytestmark = pytest.mark.model


def test_text_prefix_and_names(model_path):
    with Checkpoint(model_path) as ckpt:
        assert ckpt.text_prefix == "model.language_model."
        assert "lm_head.weight" in ckpt
        assert ckpt.shape("model.language_model.embed_tokens.weight") == (248320, 5120)


def test_nvfp4_weight_is_sane(model_path):
    with Checkpoint(model_path) as ckpt:
        module = "model.language_model.layers.0.linear_attn.in_proj_b"
        assert ckpt.is_nvfp4(module)
        w = ckpt.nvfp4(module)
        assert w.shape == (48, 5120)
        dense = nvfp4.dequantize(w, torch.float32)
        assert torch.isfinite(dense).all()
        assert 1e-4 < dense.std().item() < 1.0


def test_slices_match_full_tensor(model_path):
    with Checkpoint(model_path) as ckpt:
        name = "model.language_model.layers.0.linear_attn.in_proj_a.weight_packed"
        full = ckpt.tensor(name)
        assert torch.equal(ckpt.rows(name, 8, 20), full[8:20])
        assert torch.equal(ckpt.cols(name, 64, 128), full[:, 64:128])


def test_matches_compressed_tensors(model_path):
    helpers = pytest.importorskip("compressed_tensors.quantization.lifecycle.forward_helpers")
    with Checkpoint(model_path) as ckpt:
        w = ckpt.nvfp4("model.language_model.layers.3.self_attn.k_proj")
        values = nvfp4.unpack(w.packed)
        scale = w.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        theirs = helpers._dequantize(values, scale, global_scale=torch.tensor([w.global_scale]), dtype=torch.float32)
        assert torch.equal(nvfp4.dequantize(w, torch.float32), theirs)
