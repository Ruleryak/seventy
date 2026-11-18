"""The engine: requests in, tokens out. One process, one GPU (tensor parallel comes later)."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..checkpoint import Checkpoint
from ..config import ModelConfig, load_config
from ..model.qwen3_5 import Qwen35TextModel
from .cache import ModelCache
from .runner import ModelRunner
from .sampling import RowState, SamplingParams, TokenLogprobs, logprobs_of, sample
from .scheduler import Plan, Request, Scheduler, Status, Work


@dataclass
class EngineConfig:
    model: str
    device: str = "cuda"
    max_num_seqs: int = 8
    max_model_len: int = 262144
    block_size: int = 16
    gpu_memory_utilization: float = 0.90
    prefill_budget: int = 4096
    kv_dtype: str = "int8"
    activation_reserve_gib: float = 3.0


@dataclass
class RequestOutput:
    id: str
    new_tokens: list[int]
    finished: bool
    finish_reason: str | None
    new_logprobs: list[TokenLogprobs] = field(default_factory=list)
    prompt_logprobs: list[TokenLogprobs | None] | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0


def bytes_per_block(cfg: ModelConfig, block_size: int, kv_dtype: torch.dtype, tp: int = 1) -> int:
    t = cfg.text
    per_vec = t.head_dim * (1 if kv_dtype == torch.int8 else 2) + (4 if kv_dtype == torch.int8 else 0)
    return len(t.attention_layers) * 2 * block_size * (t.num_kv_heads // tp) * per_vec


def bytes_per_slot(cfg: ModelConfig, tp: int = 1) -> int:
    t = cfg.text
    hv = t.gdn_v_heads // tp
    conv = (2 * t.gdn_k_heads * t.gdn_k_dim + t.gdn_v_heads * t.gdn_v_dim) // tp
    return len(t.linear_layers) * (hv * t.gdn_k_dim * t.gdn_v_dim * 4 + conv * (t.conv_kernel - 1) * 2)


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        path = Path(config.model)
        self.model_config = load_config(path)
        torch.cuda.set_device(self.device)
        with Checkpoint(path) as ckpt:
            self.model = Qwen35TextModel(self.model_config, ckpt, self.device, torch.float16)
        torch.cuda.synchronize()
        kv_dtype = torch.int8 if config.kv_dtype == "int8" else torch.float16
        free, total = torch.cuda.mem_get_info(self.device)
        budget = int(total * config.gpu_memory_utilization) - (total - free)
        budget -= int(config.activation_reserve_gib * 2**30)
        slots = config.max_num_seqs
        per_block = bytes_per_block(self.model_config, config.block_size, kv_dtype)
        state = slots * bytes_per_slot(self.model_config)
        num_blocks = (budget - state) // per_block
        need = -(-config.max_model_len // config.block_size)
        if num_blocks < min(need, 64):
            raise RuntimeError(f"not enough GPU memory for the KV cache ({num_blocks} blocks)")
        num_blocks = min(num_blocks, need * slots)
        self.cache = ModelCache(self.model_config.text, num_blocks, config.block_size, slots, torch.float16,
                                self.device, kv_dtype=kv_dtype)
        self.runner = ModelRunner(self.model, self.cache, self.device)
        self.scheduler = Scheduler(num_blocks, config.block_size, slots, config.max_model_len,
                                   prefill_budget=config.prefill_budget)
        self.eos = tuple(self.model_config.text.eos_token_ids)
        self._ids = itertools.count()
        self.requests: dict[str, Request] = {}
        self.stats = {"steps": 0, "prompt_tokens": 0, "generated_tokens": 0, "preemptions": 0}

    @property
    def kv_capacity_tokens(self) -> int:
        return self.cache.num_blocks * self.cache.block_size

    def add_request(self, prompt: list[int], params: SamplingParams, request_id: str | None = None) -> str:
        params.check()
        rid = request_id or f"req-{next(self._ids)}"
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(params.seed)
        r = Request(rid, list(prompt), params, generator=gen)
        self.scheduler.add(r)
        self.requests[rid] = r
        return rid

    def abort(self, request_id: str) -> None:
        self.scheduler.abort(request_id)
        self.requests.pop(request_id, None)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def _stop_reason(self, r: Request, token: int) -> str | None:
        p = r.params
        if len(r.output) >= p.min_tokens:
            if not p.ignore_eos and token in self.eos:
                return "stop"
            if token in p.stop_token_ids:
                return "stop"
        if len(r.output) >= p.max_tokens:
            return "length"
        if r.length >= self.config.max_model_len:
            return "length"
        return None

    def _sample(self, work: list[Work], logits: torch.Tensor, outputs: dict[str, RequestOutput]) -> None:
        rows, owners, row = [], [], 0
        for w in work:
            r = w.request
            n = w.count if w.logits == "all" else (1 if w.logits == "last" else 0)
            if w.logits == "all" and r.params.prompt_logprobs is not None:
                targets = r.tokens[w.start + 1 : w.start + w.count + 1]
                k = len(targets)
                if k:
                    lps = logprobs_of(logits[row : row + k], torch.tensor(targets, device=logits.device),
                                      r.params.prompt_logprobs)
                    if not r.prompt_logprobs:
                        r.prompt_logprobs = [None]
                    r.prompt_logprobs += lps[: max(0, len(r.prompt) - len(r.prompt_logprobs))]
            r.computed = w.start + w.count
            if r.computed == r.length and n:
                rows.append(row + n - 1)
                owners.append(r)
            row += n
        if not owners:
            return
        states = [RowState(r.params, r.prompt, r.output, r.generator) for r in owners]
        tokens, lps = sample(logits[rows], states, self.eos)
        for r, t, lp in zip(owners, tokens, lps):
            r.output.append(t)
            if lp is not None:
                r.logprobs.append(lp)
            out = outputs.setdefault(r.id, RequestOutput(r.id, [], False, None))
            out.new_tokens.append(t)
            if lp is not None:
                out.new_logprobs.append(lp)
            reason = self._stop_reason(r, t)
            if reason:
                self.scheduler.finish(r, reason)

    def step(self) -> list[RequestOutput]:
        plan: Plan = self.scheduler.schedule()
        outputs: dict[str, RequestOutput] = {}
        if plan.decode:
            self._sample(plan.decode, self.runner.run(plan.decode), outputs)
        if plan.prefill:
            self.stats["prompt_tokens"] += sum(w.count for w in plan.prefill)
            self._sample(plan.prefill, self.runner.run(plan.prefill), outputs)
        self.stats["steps"] += 1
        result = []
        for rid, out in outputs.items():
            r = self.requests[rid]
            self.stats["generated_tokens"] += len(out.new_tokens)
            out.finished = r.status is Status.FINISHED
            out.finish_reason = r.finish_reason
            out.prompt_tokens, out.output_tokens = len(r.prompt), len(r.output)
            if out.finished:
                out.prompt_logprobs = r.prompt_logprobs or None
                self.requests.pop(rid, None)
            result.append(out)
        return result

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Run prompts to completion (offline use and tests). Returns the generated tokens of each prompt."""
        ids = [self.add_request(p, SamplingParams(**vars(params))) for p in prompts]
        done: dict[str, list[int]] = {i: [] for i in ids}
        while self.has_work():
            for out in self.step():
                done[out.id] += out.new_tokens
        return [done[i] for i in ids]
