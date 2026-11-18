"""Image preprocessing, multimodal positions and the vision tower against the transformers implementation."""

import json
import types

import numpy as np
import pytest
import torch

from seventy import multimodal as mm
from seventy.model.vision import patchify

SETTINGS = mm.ImageSettings()


def synthetic_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([127 + 120 * np.sin(x / 23.0), 127 + 120 * np.cos(y / 17.0), (x + y) % 256], axis=-1)
    return np.clip(base + rng.normal(0, 18, base.shape), 0, 255).astype(np.uint8)


def hf_processor():
    proc = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    pytest.importorskip("torchvision")
    return proc.Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
                                      size={"shortest_edge": 65536, "longest_edge": 16777216},
                                      image_mean=[0.5] * 3, image_std=[0.5] * 3)


@pytest.mark.parametrize("h, w", [(389, 517), (64, 900), (1200, 1600), (32, 32), (4100, 4100)])
def test_smart_resize_matches_reference(h, w):
    ref = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    assert mm.smart_resize(h, w, 32, 65536, 16777216) == ref.smart_resize(h, w, 32, 65536, 16777216)


@pytest.mark.parametrize("h, w", [(389, 517), (233, 901), (512, 512)])
def test_pixels_match_reference_processor(h, w):
    from PIL import Image

    proc = hf_processor()
    img = synthetic_image(h, w)
    ref = proc(images=[Image.fromarray(img)], return_tensors="pt")
    ours = mm.prepare_image(img, SETTINGS)
    assert list(ours.grid) == ref["image_grid_thw"][0].tolist()
    patches = patchify(torch.from_numpy(ours.pixels), 16, 2, 2)
    assert patches.shape == ref["pixel_values"].shape
    assert (patches - ref["pixel_values"]).abs().max().item() < 1e-6


def test_image_urls_and_transparency():
    from PIL import Image
    import base64
    import io

    im = Image.new("RGBA", (40, 30), (255, 0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": url}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": url}]}]
    urls = mm.image_urls(msgs)
    assert urls == [url, url]
    img = mm.load_image(urls[0], SETTINGS)
    assert img.pixels.shape[2] == 3 and img.pixels.shape[0] % 32 == 0 and img.pixels.shape[1] % 32 == 0
    assert int(img.pixels.min()) == 255
    with pytest.raises(mm.ImageError):
        mm.image_urls([{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}])
    with pytest.raises(mm.ImageError):
        mm.load_image("file:///etc/passwd", SETTINGS)


def test_rope_positions_match_reference():
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    pad = 248056
    grids = [(1, 6, 10), (1, 8, 4)]
    images = [mm.ImageInput(np.zeros((g[1] * 16, g[2] * 16, 3), np.uint8), g, str(i) * 64) for i, g in enumerate(grids)]
    ids = [1, 2, 3, 248053, pad, 248054, 7, 8, 248053, pad, 248054, 9, 10, 11]
    out, info = mm.expand_prompt(ids, images, pad, 2)
    assert len(out) == len(ids) - 2 + 15 + 8
    assert info.spans == [(4, 15), (4 + 15 + 4, 8)]
    fake = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    fake.get_vision_position_ids = lambda *a, **k: hf.Qwen3_5Model.get_vision_position_ids(fake, *a, **k)
    ids_t = torch.tensor([out])
    types_t = (ids_t == pad).int()
    ref, delta = hf.Qwen3_5Model.get_rope_index(fake, ids_t, types_t, image_grid_thw=torch.tensor(grids))
    assert np.array_equal(info.positions, ref[:, 0].numpy())
    assert info.delta == int(delta[0, 0])
    key = info.cache_key(out)
    assert key[:4] == out[:4] and all(k < 0 for k in key[4:19]) and key[4] != key[23]


def test_expand_prompt_rejects_count_mismatch():
    img = mm.ImageInput(np.zeros((32, 32, 3), np.uint8), (1, 2, 2), "0" * 64)
    with pytest.raises(mm.ImageError):
        mm.expand_prompt([1, 248056, 248056], [img], 248056, 2)


def _hf_tower(path):
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    cfg = json.loads((path / "config.json").read_text())["vision_config"]
    vc = Qwen3_5VisionConfig(**cfg)
    vc._attn_implementation = "sdpa"
    model = hf.Qwen3_5VisionModel(vc).float().eval()
    idx = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for f in sorted({v for k, v in idx.items() if k.startswith("model.visual.")}):
        with safe_open(str(path / f), "pt") as h:
            for k in h.keys():
                if k.startswith("model.visual."):
                    state[k[len("model.visual."):]] = h.get_tensor(k).float()
    model.load_state_dict(state, strict=True)
    return model


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
def test_vision_tower_matches_reference(model_path):
    from seventy.checkpoint import Checkpoint
    from seventy.config import load_config
    from seventy.model.vision import VisionTower

    cfg = load_config(model_path)
    ref_model = _hf_tower(model_path).cuda()
    with Checkpoint(model_path) as ckpt:
        tower = VisionTower(cfg.vision, ckpt, "cuda")
    for h, w, seed in [(389, 517, 1), (700, 300, 2)]:
        img = mm.prepare_image(synthetic_image(h, w, seed), SETTINGS)
        pix = torch.from_numpy(img.pixels)
        with torch.no_grad():
            ref = ref_model(patchify(pix, 16, 2, 2).cuda(), grid_thw=torch.tensor([img.grid]).cuda()).pooler_output
        ours = tower.encode(pix).float()
        assert ours.shape == ref.shape
        err = ((ours - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(ours, ref, dim=-1).min().item()
        assert err < 5e-3 and cos > 0.999, (err, cos)
