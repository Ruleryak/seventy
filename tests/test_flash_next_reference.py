"""The Qwen4-Exp text model (Qwen3.8 Flash-Next: hyper-connections, PLE n-gram embeddings, the QSA indexer, a
mixture of 512 experts) against the transformers implementation, both in FP32 on the CPU, with the first
SEVENTY_TEST_FLASH_NEXT_LAYERS layers (default 4: three Gated DeltaNet layers, the PLE layer among them, and one
attention layer) of a real checkpoint (flash_next_path). transformers' n-gram table would take about 205 GB in FP32;
its rows come from seventy's table (NGramTable) instead, which test_ngram_table_rows checks against the checkpoint.
A short prompt runs dense attention; a prompt of more than indexer_budget tokens runs the sparse selection. Needs
the `reference` extra and about 70 GiB of free host memory; with less, the tests skip."""

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

NEED_MEMORY = 70 << 30

pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    pytest.mark.skipif(available_memory() < NEED_MEMORY,
                       reason=f"needs about {NEED_MEMORY >> 30} GiB of free host memory "
                              f"({available_memory() >> 30} GiB available)"),
]

LAYERS = int(os.environ.get("SEVENTY_TEST_FLASH_NEXT_LAYERS", "4"))
PROMPT = [9707, 11, 1052, 374, 264, 1273, 315, 279, 5662, 13, 576, 3974, 13876, 38835, 34208, 916, 279,
          15678, 5562, 13, 2585, 1657, 11931, 525, 1052, 304, 279, 3409, 30, 220, 18, 15, 20, 21]


def weight(ckpt: Checkpoint, module: str) -> torch.Tensor:
    if ckpt.is_nvfp4(module):
        return nvfp4.dequantize(ckpt.nvfp4(module), torch.float32)
    return ckpt.linear_part(module).float()


class TableRows(torch.nn.Module):
    """nn.Embedding stand-in that reads rows from seventy's NGramTable."""

    def __init__(self, table):
        super().__init__()
        self.table = table
        self.weight = torch.empty(0)

    def forward(self, ids):
        return self.table.rows(ids.reshape(-1)).view(*ids.shape, self.table.dim)


def hf_reference(path, layers, tables):
    hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    raw = json.loads((path / "config.json").read_text())["text_config"]
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    raw.pop("dtype", None)
    raw.pop("mtp", None)
    cfg = Qwen4ExpTextConfig(**raw)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    with torch.device("meta"):
        model = hf.Qwen4ExpForCausalLM(cfg)
    for i, layer in enumerate(model.model.layers):
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = TableRows(tables[i])
    model.model.rotary_emb = hf.Qwen4ExpTextRotaryEmbedding(cfg)
    state = {}
    with Checkpoint(path) as ckpt:
        prefix = ckpt.text_prefix
        for name in model.state_dict():
            src = name.replace("model.", prefix, 1) if name.startswith("model.") else name
            if src.endswith("experts.gate_up_proj") or src.endswith("experts.down_proj"):
                base, kind = src.rsplit(".", 1)
                if src in ckpt:
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
                t = ckpt.tensor(src)
                state[name] = t if t.dtype == torch.int64 else t.float()
    model.load_state_dict(state, strict=True, assign=True)
    del state
    return model.eval()


@pytest.fixture(scope="module")
def models(flash_next_path):
    from seventy.model.qwen4_exp import Qwen4ExpTextModel

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cfg = load_config(flash_next_path)
    with Checkpoint(flash_next_path) as ckpt:
        ours = Qwen4ExpTextModel(cfg, ckpt, "cpu", torch.float32, layers=LAYERS)
    tables = {i: layer.ple.table for i, layer in enumerate(ours.layers) if layer.ple is not None}
    return cfg, ours, hf_reference(flash_next_path, LAYERS, tables)


def run(ours, cfg, prompt, split, blocks):
    cache = ModelCache(cfg.text, num_blocks=blocks + 2, block_size=16, num_slots=2, dtype=torch.float32, device="cpu")
    table = list(range(1, blocks + 1))
    step = StepInput.build([SequenceChunk(prompt[:split], 0, table, 1, logits="all")], 16, "cpu")
    rows = [ours.forward(step, cache)[0]]
    for i in range(split, len(prompt)):
        step = StepInput.build([SequenceChunk([prompt[i]], i, table, 1)], 16, "cpu")
        rows.append(ours.forward(step, cache)[0])
    return torch.cat(rows)


def compare(ours, ref, cfg, prompt, split, rows=None):
    with torch.inference_mode():
        expected = ref(torch.tensor([prompt])).logits[0].float()
    got = run(ours, cfg, prompt, split, -(-len(prompt) // 16))
    if rows is not None:
        got, expected = got[rows], expected[rows]
    diff = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    print(f"\nlayers={LAYERS} tokens={len(prompt)} max |logit diff| {diff:.2e} (max |logit| {scale:.1f})")
    assert diff < 2e-3 * max(scale, 1.0), (diff, scale)
    assert torch.equal(got.argmax(-1), expected.argmax(-1))


def test_ngram_table_rows(flash_next_path):
    from seventy.model.qwen4_exp import NGramTable

    with Checkpoint(flash_next_path) as ckpt:
        p = f"{ckpt.text_prefix}layers.1.ple.ple_embedding.ngram_embedding"
        table = NGramTable(ckpt, p)
        assert table.num_rows >= int(ckpt.tensor(p.replace("ngram_embedding", "ngram_heads_offsets"))[-1])
        shard = 3
        start = table.starts[shard]
        name = f"{p}.shard_{shard}.weight"
        want_w = nvfp4.Nvfp4Weight(ckpt.rows(name, 5, 9), ckpt.rows(name + "_scale", 5, 9).view(torch.uint8),
                                   table.global_scale)
        got = table.rows(torch.tensor([start + 5, start + 6, start + 7, start + 8]))
        torch.testing.assert_close(got, nvfp4.dequantize(want_w, torch.float32))


def test_short_prompt_matches_transformers(models):
    cfg, ours, ref = models
    compare(ours, ref, cfg, PROMPT, len(PROMPT) - 4)


def test_long_prompt_sparse_attention_matches_transformers(models):
    cfg, ours, ref = models
    e = cfg.text.exp
    n = e.indexer_budget + 4 * e.indexer_compress + 9
    g = torch.Generator().manual_seed(5)
    prompt = torch.randint(1000, 100000, (n,), generator=g).tolist()
    compare(ours, ref, cfg, prompt, n - 3, rows=slice(-6, None))
