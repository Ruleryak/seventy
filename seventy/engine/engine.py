"""The engine: requests in, tokens out. Rank 0 of tensor parallel runs here and drives the other ranks' worker
processes.

Images: a request's images (seventy.multimodal.PromptImages) are encoded by the vision tower on every rank just
before the first prefill chunk that reaches their tokens, and released once prefill has passed them."""

from __future__ import annotations

import faulthandler
import itertools
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp

from ..config import ModelConfig, load_config
from ..parallel import Parallel
from . import distributed
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
    tensor_parallel: int = 1
    cuda_graphs: bool = True
    prefix_cache: bool = True
    cache_entries: int = 0
    draft_model: str | None = None


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
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        tp = config.tensor_parallel
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", 0 if tp > 1 else torch.cuda.current_device())
        path = Path(config.model)
        self.model_config = load_config(path)
        torch.cuda.set_device(self.device)
        reserve = int(config.activation_reserve_gib * 2**30)
        kv_dtype = torch.int8 if config.kv_dtype == "int8" else torch.float16
        self.workers, peers = [], []
        port = distributed.free_port()
        if tp > 1:
            ctx = mp.get_context("spawn")
            for rank in range(1, tp):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker_entry, daemon=True,
                                args=(rank, tp, port, "nccl", f"cuda:{rank}", str(path), "float16",
                                      config.gpu_memory_utilization, reserve, child, None, config.draft_model))
                p.start()
                child.close()
                self.workers.append(p)
                peers.append(parent)
            distributed.init(0, tp, port, "nccl")
        from .worker import load_shard

        _, self.model = load_shard(str(path), 0, tp, self.device, torch.float16)
        self.drafter = None
        if config.draft_model:
            from ..model.dflash2 import Drafter

            self.drafter = Drafter(config.draft_model, self.model, self.device, torch.float16, par=Parallel(0, tp))
        torch.cuda.synchronize()
        budgets = [distributed.memory_budget(self.device, config.gpu_memory_utilization, reserve)]
        for conn, proc in zip(peers, self.workers):
            budgets.append(_expect(conn, "ready", proc))
        slots = config.max_num_seqs
        entries = (config.cache_entries or config.max_num_seqs) if config.prefix_cache else 0
        total_slots = slots + entries
        per_block = bytes_per_block(self.model_config, config.block_size, kv_dtype, tp)
        ring = self.drafter.ring_bytes_per_slot() if self.drafter is not None else 0
        state = (total_slots + 1) * (bytes_per_slot(self.model_config, tp) + ring)
        num_blocks = (min(budgets) - state) // per_block
        need = -(-config.max_model_len // config.block_size)
        if num_blocks < min(need, 64):
            raise RuntimeError(f"not enough GPU memory for the KV cache ({num_blocks} blocks)")
        num_blocks = min(num_blocks, need * total_slots)
        kv_name = "int8" if kv_dtype == torch.int8 else "float16"
        for conn in peers:
            conn.send(("init", num_blocks, config.block_size, total_slots, kv_name))
        self.cache = ModelCache(self.model_config.text, num_blocks, config.block_size, total_slots, torch.float16,
                                self.device, par=Parallel(0, tp), kv_dtype=kv_dtype)
        if self.drafter is not None:
            self.drafter.allocate(total_slots)
            self.cache.slot_peers.append(self.drafter)
        for conn, proc in zip(peers, self.workers):
            _expect(conn, "ok", proc)
        self.runner = ModelRunner(self.model, self.cache, self.device, peers, self.drafter)
        if config.cuda_graphs and self.drafter is not None:
            from .graphs import BUCKETS

            limit = max(1, min(slots, 64 // self.drafter.cfg.block))
            buckets = tuple(b for b in BUCKETS if b <= limit)
            if limit not in buckets:
                buckets += (limit,)
            self.runner.capture_spec(need, buckets)
        if config.cuda_graphs and self.drafter is None:
            from .graphs import BUCKETS

            buckets = tuple(b for b in BUCKETS if b <= max(1, slots))
            if slots not in buckets and slots <= 64:
                buckets += (slots,)
            self.runner.capture(need, buckets)
        self.scheduler = Scheduler(num_blocks, config.block_size, total_slots, config.max_model_len,
                                   prefill_budget=config.prefill_budget, max_running=slots, cache_entries=entries)
        self.eos = tuple(self.model_config.text.eos_token_ids)
        self._ids = itertools.count()
        self.requests: dict[str, Request] = {}
        self.stats = {"steps": 0, "prompt_tokens": 0, "generated_tokens": 0, "preemptions": 0, "cache_hits": 0,
                      "cache_hit_tokens": 0, "verify_steps": 0, "drafted_tokens": 0, "accepted_tokens": 0,
                      "images": 0}

    @property
    def kv_capacity_tokens(self) -> int:
        return self.cache.num_blocks * self.cache.block_size

    @property
    def vision(self) -> bool:
        return getattr(self.model, "vision", None) is not None

    def add_request(self, prompt: list[int], params: SamplingParams, request_id: str | None = None,
                    images=None) -> str:
        """images: seventy.multimodal.PromptImages of an expanded prompt, or None."""
        params.check()
        rid = request_id or f"req-{next(self._ids)}"
        if rid in self.requests:
            raise ValueError(f"request id {rid!r} is in use")
        prompt = list(prompt)
        if images is not None and images.images:
            if not self.vision:
                raise ValueError("this model has no vision tower; images are not supported")
            pads = {self.model_config.image_token_id, self.model_config.video_token_id}
            for off, n in images.spans:
                if off < 0 or off + n > len(prompt) or len(set(prompt[off : off + n])) != 1 or prompt[off] not in pads:
                    raise ValueError("image spans do not match the prompt's image and video tokens")
            if images.positions.shape != (3, len(prompt)):
                raise ValueError("image positions do not match the prompt")
        else:
            images = None
        gen = torch.Generator()
        gen.manual_seed(params.seed if params.seed is not None else torch.randint(0, 2**62, ()).item())
        r = Request(rid, prompt, params, generator=gen, images=images,
                    cache_key=images.cache_key(prompt) if images is not None else None)
        self.scheduler.add(r)
        self.requests[rid] = r
        return rid

    def abort(self, request_id: str) -> None:
        r = self.requests.get(request_id)
        self.scheduler.abort(request_id)
        if r is not None:
            self._release_images(r, everything=True)
        self.requests.pop(request_id, None)

    def _encode_images(self, work: list[Work]) -> None:
        for w in work:
            r = w.request
            for _, i, _, _ in r.image_rows(w.start, w.count):
                key = f"{r.id}/{i}"
                if key not in self.runner.images:
                    self.runner.encode(key, r.images.images[i].pixels)
                    self.stats["images"] += 1

    def _release_images(self, r: Request, everything: bool = False) -> None:
        if r.images is None:
            return
        keys = [f"{r.id}/{i}" for i, (off, n) in enumerate(r.images.spans)
                if (everything or r.status is Status.FINISHED or r.computed >= off + n)
                and f"{r.id}/{i}" in self.runner.images]
        self.runner.release(keys)

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
        if plan.copies:
            self.runner.copy(plan.copies)
        if plan.decode:
            if self.drafter is not None:
                self._spec_decode(plan.decode, outputs)
            else:
                self._sample(plan.decode, self.runner.run_decode(plan.decode), outputs)
        if plan.prefill:
            self.stats["prompt_tokens"] += sum(w.count for w in plan.prefill)
            self._encode_images(plan.prefill)
            logits = self.runner.run(plan.prefill, capture=self.drafter is not None)
            if self.drafter is not None:
                self._prefill_context(plan.prefill)
            self._sample(plan.prefill, logits, outputs)
            for w in plan.prefill:
                self._release_images(w.request)
            if self.drafter is not None:
                self._draft([w.request for w in plan.prefill
                             if w.request.status is Status.RUNNING and w.request.computed == w.request.length - 1])
        self.stats["steps"] += 1
        self.stats["cache_hits"] = self.scheduler.cache.hits
        self.stats["cache_hit_tokens"] = self.scheduler.cache.hit_tokens
        self.stats["preemptions"] = self.scheduler.preempted
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

    def _prefill_context(self, work: list[Work]) -> None:
        """Drafter context for the prefill rows within the drafter's window of the sequence end."""
        window = self.drafter.cfg.window
        rows, positions, slots, row = [], [], [], 0
        for w in work:
            first = max(w.start, w.request.length - window)
            for i in range(w.count):
                if w.start + i >= first:
                    rows.append(row + i)
                    positions.append(w.start + i)
                    slots.append(w.request.slot)
            row += w.count
        self.runner.context(rows, positions, slots)

    def _draft(self, requests: list[Request]) -> None:
        if not requests:
            return
        d = self.runner.draft([r.tokens[-1] for r in requests], [r.computed for r in requests],
                              [r.slot for r in requests], [r.params.temperature for r in requests],
                              [r.generator for r in requests])
        tokens = d.tokens.tolist()
        probs, cands = d.probs.cpu(), d.candidates.cpu()
        for i, r in enumerate(requests):
            r.draft, r.draft_q, r.draft_cand = tokens[i], probs[i], cands[i]
        self.stats["drafted_tokens"] += sum(len(t) for t in tokens)

    def _stops(self, r: Request) -> tuple[list[int], int]:
        """A request's stop tokens and the most tokens it may still emit."""
        p = r.params
        stop = ([] if p.ignore_eos else list(self.eos)) + [t for t in p.stop_token_ids if t not in self.eos]
        limit = min(p.max_tokens - len(r.output), self.config.max_model_len - r.length)
        return stop, max(1, limit)

    def _spec_rows(self, work: list[Work]) -> list[dict] | None:
        from .spec import STOP_SLOTS
        from .spec_graphs import eligible

        g = self.runner.spec_graphs
        k = self.drafter.cfg.block - 1
        if g is None or g.bucket(len(work)) is None:
            return None
        rows = []
        for w in work:
            r = w.request
            p = r.params
            stop, limit = self._stops(r)
            if len(r.draft) != k or not eligible(p) or len(r.output) < p.min_tokens or len(stop) > STOP_SLOTS:
                return None
            row = {"tokens": [r.tokens[-1]] + r.draft, "start": w.start, "delta": r.rope_delta, "slot": r.slot,
                   "table": r.blocks,
                   "temp": p.temperature, "top_k": p.top_k if p.temperature > 0 else 1, "top_p": p.top_p,
                   "min_p": p.min_p, "q": r.draft_q, "cand": r.draft_cand, "stop": stop, "limit": limit}
            if p.temperature > 0:
                gen = r.generator
                row["u_accept"] = torch.rand(k, generator=gen)
                row["u_residual"] = torch.rand(k + 1, generator=gen)
                row["gumbel"] = -torch.log(-torch.log(torch.rand(k, g.top, generator=gen).clamp(1e-20, 1.0)))
            rows.append(row)
        return rows

    def _spec_decode(self, work: list[Work], outputs: dict[str, RequestOutput]) -> None:
        from .spec import accept, cut_host

        rows = self._spec_rows(work)
        if rows is not None:
            out = self.runner.spec_step(rows)
            ns, news = out["n"].tolist(), out["new"].tolist()
            drafts, probs, cands = out["drafts"].tolist(), out["probs"].cpu(), out["cand"].cpu()
            self.stats["verify_steps"] += len(work)
            for i, w in enumerate(work):
                r, n = w.request, ns[i]
                self.stats["accepted_tokens"] += n
                r.computed = w.start + n + 1
                r.draft, r.draft_q, r.draft_cand = drafts[i], probs[i], cands[i]
                o = outputs.setdefault(r.id, RequestOutput(r.id, [], False, None))
                reason = None
                for j, t in enumerate(news[i][: n + 1]):
                    r.output.append(t)
                    o.new_tokens.append(t)
                    reason = self._stop_reason(r, t)
                    if reason:
                        if j < n:
                            r.cacheable = False
                        break
                if reason:
                    self.scheduler.finish(r, reason)
            self.stats["drafted_tokens"] += sum(len(d) for d in drafts)
            return
        logits = self.runner.verify(work)
        lens, rows, positions, slots, results, row = [], [], [], [], [], 0
        for w in work:
            r = w.request
            n_rows = w.count if w.logits == "all" else 1
            L = logits[row : row + n_rows]
            state = RowState(r.params, r.prompt, r.output, r.generator)
            if r.draft and n_rows == 1 + len(r.draft):
                n, new = accept(L, r.draft, r.draft_q.to(L.device), r.draft_cand.to(L.device), state, self.eos)
            else:
                n, new = 0, sample(L[-1:], [state], self.eos)[0]
            stop, limit = self._stops(r)
            n = cut_host(n, new, set(stop), limit)
            new = new[: n + 1]
            results.append((w, n, new, L))
            lens.append(n + 1)
            rows += list(range(row, row + n + 1))
            positions += list(range(w.start, w.start + n + 1))
            slots += [r.slot] * (n + 1)
            row += n_rows
        self.runner.accept(lens, rows, positions, slots)
        self.stats["verify_steps"] += len(work)
        survivors = []
        for w, n, new, L in results:
            r = w.request
            self.stats["accepted_tokens"] += n
            r.computed = w.start + n + 1
            r.draft, r.draft_q, r.draft_cand = [], None, None
            out = outputs.setdefault(r.id, RequestOutput(r.id, [], False, None))
            lp_rows = logprobs_of(L[: len(new)], torch.tensor(new, device=L.device), r.params.logprobs or 0) \
                if r.params.logprobs is not None else [None] * len(new)
            reason = None
            for i, (t, lp) in enumerate(zip(new, lp_rows)):
                r.output.append(t)
                out.new_tokens.append(t)
                if lp is not None:
                    lp.top = lp.top[: r.params.logprobs]
                    r.logprobs.append(lp)
                    out.new_logprobs.append(lp)
                reason = self._stop_reason(r, t)
                if reason:
                    if i < n:
                        r.cacheable = False
                    break
            if reason:
                self.scheduler.finish(r, reason)
            else:
                survivors.append(r)
        self._draft(survivors)

    def close(self) -> None:
        """Stop the workers. Every rank tears its process group down at the same time (NCCL needs that)."""
        self.runner.close()
        self.runner.graphs = None
        self.runner.spec_graphs = None
        distributed.shutdown()
        for p in self.workers:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[list[int]]:
        """Run prompts to completion (offline use and tests). Returns the generated tokens of each prompt."""
        ids = [self.add_request(p, SamplingParams(**vars(params))) for p in prompts]
        done: dict[str, list[int]] = {i: [] for i in ids}
        while self.has_work():
            for out in self.step():
                done[out.id] += out.new_tokens
        return [done[i] for i in ids]


def _worker_entry(*args):
    from .worker import worker_main

    worker_main(*args)


def _expect(conn, kind: str, proc=None, timeout: float = 900.0):
    waited = 0.0
    while not conn.poll(5.0):
        waited += 5.0
        if proc is not None and not proc.is_alive():
            raise RuntimeError(f"tensor-parallel worker {proc.name} exited with code {proc.exitcode}")
        if waited >= timeout:
            raise RuntimeError(f"no {kind!r} message from a tensor-parallel worker after {timeout:.0f} s")
    try:
        msg = conn.recv()
    except EOFError:
        raise RuntimeError("a tensor-parallel worker closed its pipe") from None
    if msg[0] == "error":
        raise RuntimeError(f"a tensor-parallel worker failed:\n{msg[1]}")
    if msg[0] != kind:
        raise RuntimeError(f"unexpected worker message {msg[0]!r}, wanted {kind!r}")
    return msg[1]
