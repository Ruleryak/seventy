"""Compare seventy's prompt log-probabilities with a running OpenAI-compatible server on the same tokens.

The server must support `prompt_logprobs` in /v1/completions (vLLM-style servers do).

    python bench/compare_logprobs.py --model /path/to/checkpoint --server http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import torch

DEFAULT_TEXT = (
    "The history of the telescope begins in the Netherlands in 1608, when spectacle makers applied for patents on "
    "a device that made distant objects appear closer. Within a year Galileo Galilei had built his own instrument "
    "and turned it to the night sky, where he saw the phases of Venus, the moons of Jupiter and the craters of the "
    "Moon. These observations supported the idea that the Earth moves around the Sun. Later designs used curved "
    "mirrors instead of lenses, because a mirror does not split white light into colours. Isaac Newton built the "
    "first practical reflecting telescope in 1668. Today the largest telescopes have mirrors made of many "
    "hexagonal segments, and some observe in wavelengths that the human eye cannot see, such as infrared light "
    "and radio waves.\n\ndef mean(values):\n    total = 0\n    for v in values:\n        total += v\n"
    "    return total / len(values)\n\nprint(mean([3, 5, 10]))\n"
)


def server_logprobs(server: str, name: str | None, ids: list[int]) -> tuple[list[float], list[int]]:
    if name is None:
        with urllib.request.urlopen(f"{server}/v1/models", timeout=30) as r:
            name = json.load(r)["data"][0]["id"]
    body = {"model": name, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    req = urllib.request.Request(f"{server}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        entries = json.load(r)["choices"][0]["prompt_logprobs"]
    actual, top = [], []
    for i in range(1, len(ids)):
        entry = {int(k): v for k, v in entries[i].items()}
        actual.append(entry[ids[i]]["logprob"])
        top.append(min(entry, key=lambda k: entry[k].get("rank", 1 << 30)))
    return actual, top


def our_logprobs(model_path: Path, ids: list[int], device: str, dtype: torch.dtype, layers: int | None):
    from seventy.checkpoint import Checkpoint
    from seventy.config import load_config
    from seventy.engine.cache import ModelCache
    from seventy.engine.step import SequenceChunk, StepInput
    from seventy.model.qwen3_5 import Qwen35TextModel

    cfg = load_config(model_path)
    t0 = time.time()
    with Checkpoint(model_path) as ckpt:
        model = Qwen35TextModel(cfg, ckpt, device, dtype, layers=layers)
    load_s = time.time() - t0
    blocks = -(-len(ids) // 16)
    cache = ModelCache(cfg.text, num_blocks=blocks, block_size=16, num_slots=1, dtype=dtype, device=device)
    step = StepInput.build([SequenceChunk(ids, 0, list(range(blocks)), 0, logits="all")], 16, device)
    t0 = time.time()
    logits, _ = model.forward(step, cache)
    if device != "cpu":
        torch.cuda.synchronize()
    run_s = time.time() - t0
    lp = torch.log_softmax(logits.float(), dim=-1).cpu()
    target = torch.tensor(ids[1:])
    actual = lp[:-1].gather(1, target[:, None])[:, 0].tolist()
    return actual, lp[:-1].argmax(-1).tolist(), load_s, run_s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--server", required=True)
    ap.add_argument("--served-name")
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=("float32", "float16"), default=None)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--json", type=Path, help="write per-token results here")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    text = a.text_file.read_text() if a.text_file else DEFAULT_TEXT
    ids = Tokenizer.from_file(str(a.model / "tokenizer.json")).encode(text).ids[: a.max_tokens]
    dtype = getattr(torch, a.dtype) if a.dtype else (torch.float32 if a.device == "cpu" else torch.float16)
    ref, ref_top = server_logprobs(a.server.rstrip("/"), a.served_name, ids)
    ours, our_top, load_s, run_s = our_logprobs(a.model, ids, a.device, dtype, a.layers)
    diffs = [abs(x - y) for x, y in zip(ours, ref)]
    agree = sum(x == y for x, y in zip(our_top, ref_top)) / len(ref_top)
    print(f"{len(ids)} tokens, seventy on {a.device} {str(dtype).split('.')[-1]}: load {load_s:.0f} s, step {run_s:.1f} s")
    print(f"mean |dlogprob| {statistics.mean(diffs):.4f}, median {statistics.median(diffs):.4f}, "
          f"max {max(diffs):.4f}; top-1 agreement {agree:.1%}")
    print(f"mean logprob: seventy {statistics.mean(ours):.4f}, server {statistics.mean(ref):.4f}")
    if a.json:
        a.json.write_text(json.dumps({"ids": ids, "ours": ours, "server": ref, "our_top": our_top,
                                      "server_top": ref_top}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
