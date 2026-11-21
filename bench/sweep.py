"""Measure the decode and prefill speed of a running seventy server and write a results file for bench/report.py.

For each context length, the sweep makes one prompt per stream of the largest cell from Python source code (the
standard library of the Python that runs this script). It sends each prompt once with max_tokens 1: this is the
cold prefill, and the prompt is then in the prefix cache. Then, for each number of streams N and each run, it
sends the first N prompts at the same time and times the output tokens of each request (ignore_eos, thinking on,
temperature 0.6, top_p 0.95). Per stream is the output rate of one request from its first to its last token;
total is the output rate of all requests together while all of them run.

    python bench/sweep.py --url http://127.0.0.1:8000 --release 1.1.0 --out bench/results/1.1.0.json

The server must have room in its KV cache for all streams of the largest cell: streams x (context + output tokens).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import statistics
import subprocess
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, path: str, body: dict, key: str | None, timeout: float = 600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def get(url: str, path: str, key: str | None) -> dict:
    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


class Prompts:
    """Prompt bodies of a given token count, cut from the Python standard library sources in a seeded order."""

    def __init__(self, url: str, key: str | None):
        self.url, self.key = url, key
        root = Path(sysconfig.get_paths()["stdlib"])
        self.files = sorted(p for p in root.rglob("*.py") if 8_000 < p.stat().st_size < 300_000)
        if len(self.files) < 100:
            raise SystemExit(f"too few source files in {root}")

    def body(self, tokens: int, seed: int) -> str:
        r = random.Random(seed)
        text, need = [], tokens * 6
        while sum(len(t) for t in text) < need:
            src = r.choice(self.files).read_text(errors="replace")
            start = r.randrange(0, max(1, len(src) - 4000))
            text.append(src[start:])
        with post(self.url, "/tokenize", {"prompt": "".join(text)}, self.key) as resp:
            ids = json.load(resp)["tokens"][:tokens]
        with post(self.url, "/detokenize", {"tokens": ids}, self.key) as resp:
            return json.load(resp)["prompt"]


def request(url: str, key: str | None, model: str, messages: list, max_tokens: int, think: bool) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0.6,
            "top_p": 0.95, "stream_options": {"include_usage": True}, "ignore_eos": max_tokens > 1,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    first = last = None
    usage = {}
    pieces = []
    with post(url, "/v1/chat/completions", body, key, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if text:
                    last = time.time()
                    first = first or last
                    pieces.append((last, len(text)))
    n = usage.get("completion_tokens", 0)
    return {"t0": t0, "first": first, "last": last, "out": n, "prompt": usage.get("prompt_tokens", 0),
            "ttft": first - t0 if first else None, "pieces": pieces,
            "rate": (n - 1) / (last - first) if first and last and last > first and n > 1 else None}


def total_rate(res: list[dict]) -> float | None:
    """Output tokens per second of all streams together while every stream runs (from the first token of the last
    stream to start to the last token of the first stream to end). A stream's tokens in that window are estimated
    from its share of the output text, because a streamed chunk can hold several tokens."""
    win0, win1 = max(r["first"] for r in res), min(r["last"] for r in res)
    if win1 <= win0:
        return None
    tokens = 0.0
    for r in res:
        chars = sum(c for _, c in r["pieces"])
        inside = sum(c for t, c in r["pieces"] if win0 < t <= win1)
        tokens += r["out"] * inside / chars if chars else 0.0
    return tokens / (win1 - win0)


def messages(body: str, stream: int, seed: int) -> list:
    return [{"role": "user", "content": f"[stream {stream}, run {seed}] Here is part of a code base:\n\n{body}\n\n"
             "Explain in detail what this code does, section by section, and suggest concrete improvements."}]


def gpus() -> str:
    try:
        names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                               text=True, check=True).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        return ""
    names = [n.strip() for n in names if n.strip()]
    return f"{len(names)} x {names[0]}" if names else ""


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write(results: dict, out: Path) -> None:
    """The results file with one decode or prefill cell per line."""
    cells = {k: results.pop(k) for k in ("decode", "prefill")}
    text = json.dumps(results, indent=2)[:-2]
    for k, v in cells.items():
        text += f',\n  "{k}": [\n' + ",\n".join("    " + json.dumps(c) for c in v) + "\n  ]"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--contexts", default="2000,32000,100000", help="prompt tokens, comma-separated")
    ap.add_argument("--streams", default="1,4,8", help="requests at the same time, comma-separated")
    ap.add_argument("--repeats", type=int, default=2, help="runs of each cell")
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=0, help="change it for prompts that are not in the cache yet")
    ap.add_argument("--release", required=True)
    ap.add_argument("--name", help="name in the charts (default: the release)")
    ap.add_argument("--note", default="")
    ap.add_argument("--commit", default=None, help="default: the git commit of this folder")
    ap.add_argument("--gpus", default=None, help='for example "4 x Tesla V100-SXM2-32GB, NVLink"')
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    key = os.environ.get("SEVENTY_API_KEY")
    url = args.url.rstrip("/")
    think = args.think == "on"

    model = get(url, "/v1/models", key)["data"][0]["id"]
    try:
        info = get(url, "/info", key)
    except urllib.error.HTTPError:
        info = {}
    prompts = Prompts(url, key)
    decode, prefill = {}, {}
    started = None
    counts = [int(x) for x in args.streams.split(",")]
    for ctx in (int(x) for x in args.contexts.split(",")):
        msgs = [messages(prompts.body(ctx, 100_000 * args.seed + 7919 * i + ctx), i, args.seed)
                for i in range(max(counts))]
        started = started or time.time()
        for m in msgs:
            warm = request(url, key, model, m, 1, think)
            if warm["ttft"]:
                prefill.setdefault(ctx, []).append(round(warm["prompt"] / warm["ttft"]))
        print(f"context {ctx:>7,}: cold prefill {statistics.mean(prefill.get(ctx, [0])):,.0f} tok/s", flush=True)
        for streams in counts:
            for rep in range(args.repeats):
                res: list = [None] * streams
                threads = [threading.Thread(target=lambda j=j: res.__setitem__(
                    j, request(url, key, model, msgs[j], args.output_tokens, think))) for j in range(streams)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                rates = [r["rate"] for r in res if r["rate"]]
                total = total_rate(res)
                cell = decode.setdefault((ctx, streams), {"per_stream": [], "total": [], "ttft_cached": []})
                if rates:
                    cell["per_stream"].append(round(statistics.mean(rates), 1))
                if total:
                    cell["total"].append(round(total, 1))
                cell["ttft_cached"].append(round(statistics.mean(r["ttft"] for r in res if r["ttft"]), 3))
                print(f"context {ctx:>7,} streams {streams} run {rep + 1}: "
                      f"{cell['per_stream'][-1] if rates else 0:6.1f} tok/s per stream, {total or 0:6.1f} in total, "
                      f"first token {cell['ttft_cached'][-1]:.2f} s", flush=True)

    draft = info.get("draft_name") or ("yes" if info.get("draft") else "none")
    results = {
        "format": 1,
        "method": 2,
        "release": args.release,
        "name": args.name or args.release,
        "note": args.note,
        "commit": commit() if args.commit is None else args.commit,
        "date": datetime.datetime.fromtimestamp(started or time.time(), datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%MZ"),
        "setup": {
            "gpus": args.gpus or gpus(),
            "tensor_parallel": info.get("tensor_parallel"),
            "model": info.get("model_name") or Path(info.get("model", model)).name,
            "drafter": draft,
            "kv_cache": info.get("kv_dtype", ""),
            "thinking": think,
            "output_tokens": args.output_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "prompt": "Python standard library source code",
        },
        "decode": [dict(context=c, streams=s, **v) for (c, s), v in sorted(decode.items())],
        "prefill": [{"context": c, "tokens_per_s": v} for c, v in sorted(prefill.items())],
    }
    write(results, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
