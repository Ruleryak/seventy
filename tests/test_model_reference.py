"""The text model against the transformers implementation, both in FP32 on the CPU, with the first
SEVENTY_TEST_LAYERS layers (default 4) of the real checkpoint. Needs the `reference` extra and about 50 GiB of free
host memory (both models hold the FP32 embedding and output head); with less, the tests skip."""

import os

import pytest
import torch

from conftest import available_memory
from seventy import nvfp4
from seventy.checkpoint import Checkpoint
from seventy.config import load_config
from seventy.engine.cache import ModelCache
from seventy.engine.step import SequenceChunk, StepInput
from seventy.model.qwen3_5 import Qwen35TextModel

NEED_MEMORY = 50 << 30

pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    pytest.mark.skipif(available_memory() < NEED_MEMORY,
                       reason=f"needs about {NEED_MEMORY >> 30} GiB of free host memory "
                              f"({available_memory() >> 30} GiB available)"),
]

LAYERS = int(os.environ.get("SEVENTY_TEST_LAYERS", "4"))
PROMPT = [9707, 11, 1052, 374, 264, 1273, 315, 279, 5662, 13, 576, 3974, 13876, 38835, 34208, 916, 279,
          15678, 5562, 13, 2585, 1657, 11931, 525, 1052, 304, 279, 3409, 30, 220, 18, 15, 20, 21]


def hf_reference(path, layers):
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    import json

    raw = json.loads((path / "config.json").read_text())["text_config"]
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    raw.pop("dtype", None)
    cfg = Qwen3_5TextConfig(**raw)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.Qwen3_5ForCausalLM(cfg)
    model.to_empty(device="cpu")
    model.model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(cfg)
    state = {}
    with Checkpoint(path) as ckpt:
        prefix = ckpt.text_prefix
        for name in model.state_dict():
            src = name.replace("model.", prefix, 1) if name.startswith("model.") else name
            module = src[: -len(".weight")] if src.endswith(".weight") else None
            if module and ckpt.is_nvfp4(module):
                state[name] = nvfp4.dequantize(ckpt.nvfp4(module), torch.float32)
            else:
                state[name] = ckpt.tensor(src).float()
    model.load_state_dict(state, strict=True)
    return model.eval()


@pytest.fixture(scope="module")
def models(model_path):
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cfg = load_config(model_path)
    with Checkpoint(model_path) as ckpt:
        ours = Qwen35TextModel(cfg, ckpt, "cpu", torch.float32, layers=LAYERS, keep_dense=True)
    return cfg, ours, hf_reference(model_path, LAYERS)


def test_prefill_and_decode_match_transformers(models):
    cfg, ours, ref = models
    with torch.inference_mode():
        expected = ref(torch.tensor([PROMPT])).logits[0].float()
    cache = ModelCache(cfg.text, num_blocks=8, block_size=16, num_slots=2, dtype=torch.float32, device="cpu")
    split = len(PROMPT) - 4
    table = [5, 1, 6]
    step = StepInput.build([SequenceChunk(PROMPT[:split], 0, table, 1, logits="all")], 16, "cpu")
    prefill, _ = ours.forward(step, cache)
    rows = [prefill]
    for i in range(split, len(PROMPT)):
        step = StepInput.build([SequenceChunk([PROMPT[i]], i, table, 1)], 16, "cpu")
        rows.append(ours.forward(step, cache)[0])
    got = torch.cat(rows)
    assert got.shape == expected.shape
    diff = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    print(f"\nlayers={LAYERS} max |logit diff| {diff:.2e} (max |logit| {scale:.1f})")
    assert diff < 2e-3 * max(scale, 1.0), (diff, scale)
    assert torch.equal(got.argmax(-1), expected.argmax(-1))


def test_batched_sequences_are_independent(models):
    cfg, ours, _ = models
    a, b = PROMPT[:20], PROMPT[5:31]
    cache = ModelCache(cfg.text, num_blocks=8, block_size=16, num_slots=3, dtype=torch.float32, device="cpu")
    step = StepInput.build([SequenceChunk(a, 0, [0, 1], 0), SequenceChunk(b, 0, [2, 3], 2)], 16, "cpu")
    together, _ = ours.forward(step, cache)
    alone = []
    for seq in (a, b):
        c = ModelCache(cfg.text, num_blocks=4, block_size=16, num_slots=1, dtype=torch.float32, device="cpu")
        alone.append(ours.forward(StepInput.build([SequenceChunk(seq, 0, [0, 1], 0)], 16, "cpu"), c)[0])
    torch.testing.assert_close(together, torch.cat(alone), rtol=1e-4, atol=1e-3)
