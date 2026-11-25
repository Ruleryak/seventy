"""The Qwen3.5-MoE text model (mixture-of-experts MLP, ModelOpt NVFP4 and FP8 weights) against the transformers
implementation, both in FP32 on the CPU, with the first SEVENTY_TEST_MOE_LAYERS layers (default 4) of a real
checkpoint (moe_model_path). Both models get the same dequantized weights. Needs the `reference` extra and about
40 GiB of free host memory; with less, the tests skip."""

import json
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
from seventy.parallel import Parallel

NEED_MEMORY = 40 << 30

pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    pytest.mark.skipif(available_memory() < NEED_MEMORY,
                       reason=f"needs about {NEED_MEMORY >> 30} GiB of free host memory "
                              f"({available_memory() >> 30} GiB available)"),
]

LAYERS = int(os.environ.get("SEVENTY_TEST_MOE_LAYERS", "4"))
PROMPT = [9707, 11, 1052, 374, 264, 1273, 315, 279, 5662, 13, 576, 3974, 13876, 38835, 34208, 916, 279,
          15678, 5562, 13, 2585, 1657, 11931, 525, 1052, 304, 279, 3409, 30, 220, 18, 15, 20, 21]


def weight(ckpt: Checkpoint, module: str) -> torch.Tensor:
    """A linear module's weight in FP32, from any format the loader reads."""
    if ckpt.is_nvfp4(module):
        return nvfp4.dequantize(ckpt.nvfp4(module), torch.float32)
    return ckpt.linear_part(module).float()


def hf_reference(path, layers):
    hf = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    raw = json.loads((path / "config.json").read_text())["text_config"]
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    raw.pop("dtype", None)
    cfg = Qwen3_5MoeTextConfig(**raw)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    with torch.device("meta"):
        model = hf.Qwen3_5MoeForCausalLM(cfg)
    model.to_empty(device="cpu")
    model.model.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(cfg)
    state = {}
    with Checkpoint(path) as ckpt:
        prefix = ckpt.text_prefix
        for name in model.state_dict():
            src = name.replace("model.", prefix, 1) if name.startswith("model.") else name
            if src.endswith("experts.gate_up_proj") or src.endswith("experts.down_proj"):
                base, kind = src.rsplit(".", 1)
                if f"{src}" in ckpt:
                    state[name] = ckpt.tensor(src).float()
                elif kind == "gate_up_proj":
                    state[name] = torch.stack([torch.cat((weight(ckpt, f"{base}.{e}.gate_proj"),
                                                          weight(ckpt, f"{base}.{e}.up_proj")))
                                               for e in range(cfg.num_experts)])
                else:
                    state[name] = torch.stack([weight(ckpt, f"{base}.{e}.down_proj") for e in range(cfg.num_experts)])
                continue
            module = src[: -len(".weight")] if src.endswith(".weight") else None
            if module and ckpt.kind(module) != "dense":
                state[name] = weight(ckpt, module)
            else:
                state[name] = ckpt.tensor(src).float()
    model.load_state_dict(state, strict=True)
    return model.eval()


@pytest.fixture(scope="module")
def models(moe_model_path):
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cfg = load_config(moe_model_path)
    with Checkpoint(moe_model_path) as ckpt:
        ours = Qwen35TextModel(cfg, ckpt, "cpu", torch.float32, layers=LAYERS, keep_dense=True)
    return cfg, ours, hf_reference(moe_model_path, LAYERS)


def run(ours, cfg):
    cache = ModelCache(cfg.text, num_blocks=8, block_size=16, num_slots=2, dtype=torch.float32, device="cpu",
                       par=ours.par)
    split = len(PROMPT) - 4
    table = [5, 1, 6]
    step = StepInput.build([SequenceChunk(PROMPT[:split], 0, table, 1, logits="all")], 16, "cpu")
    rows = [ours.forward(step, cache)[0]]
    for i in range(split, len(PROMPT)):
        step = StepInput.build([SequenceChunk([PROMPT[i]], i, table, 1)], 16, "cpu")
        rows.append(ours.forward(step, cache)[0])
    return torch.cat(rows)


def test_prefill_and_decode_match_transformers(models):
    cfg, ours, ref = models
    with torch.inference_mode():
        expected = ref(torch.tensor([PROMPT])).logits[0].float()
    got = run(ours, cfg)
    assert got.shape == expected.shape
    diff = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    print(f"\nlayers={LAYERS} max |logit diff| {diff:.2e} (max |logit| {scale:.1f})")
    assert diff < 2e-3 * max(scale, 1.0), (diff, scale)
    assert torch.equal(got.argmax(-1), expected.argmax(-1))
