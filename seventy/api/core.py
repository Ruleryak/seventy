"""The engine process and the API process's client for it.

The engine process (rank 0 of tensor parallel, which starts the other ranks) runs the step loop. It takes
requests and aborts over one ZeroMQ socket and sends every step's outputs, with a snapshot of its counters, over
another. The API process tokenizes, renders templates, detokenizes and parses, so none of that delays a step.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict

import zmq
import zmq.asyncio

from ..engine.engine import EngineConfig, RequestOutput
from ..engine.sampling import SamplingParams


def engine_process_main(config: dict, req_addr: str, out_addr: str, ready) -> None:
    """The engine process. It ignores SIGINT: the API process stops it with a "shutdown" message, so the
    tensor-parallel ranks tear down together. It also stops when the API process is gone."""
    import signal

    from ..engine.engine import Engine, display_name

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = os.getppid()

    try:
        engine = Engine(EngineConfig(**config))
    except Exception:
        ready.send(("error", traceback.format_exc()))
        raise
    c = engine.config
    ready.send(("ready", {"kv_tokens": engine.kv_capacity_tokens, "pid": os.getpid(), "vision": engine.vision,
                          "draft": engine.drafter is not None, "model_name": display_name(c.model),
                          "draft_name": display_name(c.draft_model) if engine.drafter is not None else None,
                          "tensor_parallel": c.tensor_parallel,
                          "max_model_len": c.max_model_len, "max_num_seqs": c.max_num_seqs,
                          "kv_dtype": c.kv_dtype, "memory": engine.memory}))
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.connect(req_addr)
    push = ctx.socket(zmq.PUSH)
    push.connect(out_addr)
    started = time.time()
    counters = {"requests": 0, "finished": 0, "aborted": 0, "errors": 0}
    step_ms = 0.0

    def handle(msg) -> bool:
        kind = msg[0]
        if kind == "add":
            _, rid, prompt, params, images = msg
            try:
                engine.add_request(prompt, SamplingParams(**params), request_id=rid, images=images)
                counters["requests"] += 1
            except ValueError as e:
                counters["errors"] += 1
                push.send(pickle.dumps(("error", rid, str(e))))
        elif kind == "abort":
            engine.abort(msg[1])
            counters["aborted"] += 1
        elif kind == "shutdown":
            return False
        return True

    def snapshot() -> dict:
        s = engine.scheduler
        return {
            **counters, **engine.stats,
            "running": len(s.running), "waiting": len(s.waiting),
            "kv_blocks_total": s.blocks.total, "kv_blocks_free": len(s.blocks.free),
            "block_size": engine.cache.block_size, "step_ms": step_ms, "uptime_s": time.time() - started,
        }

    alive = True
    failed = False
    try:
        while alive:
            if not engine.has_work():
                if pull.poll(1000) == 0:
                    if os.getppid() != parent:
                        print("seventy: the API process is gone; stopping the engine", flush=True)
                        break
                    push.send(pickle.dumps(("stats", snapshot())))
                    continue
            while alive and pull.poll(0):
                alive = handle(pickle.loads(pull.recv()))
            if not alive or not engine.has_work():
                continue
            t0 = time.perf_counter()
            outputs = engine.step()
            step_ms = 0.9 * step_ms + 0.1 * (time.perf_counter() - t0) * 1000 if step_ms else (time.perf_counter() - t0) * 1000
            counters["finished"] += sum(1 for o in outputs if o.finished)
            push.send(pickle.dumps(("out", outputs, snapshot())))
    except BaseException:
        failed = True
        traceback.print_exc()
        raise
    finally:
        print("seventy: stopping the engine", flush=True)
        _exit_after(60 if failed else 300, 70)
        engine.close()
        print("seventy: engine stopped", flush=True)


def _exit_after(seconds: float, code: int) -> None:
    """End this process if it still runs after `seconds`: after a failed step the workers may wait in a collective
    that never completes, and the shutdown would wait for them. The workers and the API process end when this
    process is gone."""
    import threading

    timer = threading.Timer(seconds, lambda: (print("seventy: the engine did not stop; exiting", flush=True),
                                              os._exit(code)))
    timer.daemon = True
    timer.start()


class Histogram:
    """A Prometheus histogram: cumulative bucket counts, sum and count."""

    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for i, b in enumerate(self.buckets):
            if value <= b:
                self.counts[i] += 1

    def lines(self, name: str, help_text: str) -> list[str]:
        out = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        out += [f'{name}_bucket{{le="{b}"}} {c}' for b, c in zip(self.buckets, self.counts)]
        out += [f'{name}_bucket{{le="+Inf"}} {self.count}', f"{name}_sum {self.sum}", f"{name}_count {self.count}"]
        return out


class EngineClient:
    """Starts the engine process and streams outputs of requests to the API process's coroutines."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.queues: dict[str, asyncio.Queue] = {}
        self.stats: dict = {}
        self.info: dict = {}
        self.process = None
        self.dead: str | None = None
        self.ttft = Histogram((0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0))
        self.latency = Histogram((1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0))

    async def start(self) -> None:
        import torch.multiprocessing as mp

        token = uuid.uuid4().hex[:12]
        base = os.environ.get("SEVENTY_IPC_DIR") or tempfile.gettempdir()
        self.req_addr = f"ipc://{base}/seventy-{token}-req"
        self.out_addr = f"ipc://{base}/seventy-{token}-out"
        self.ctx = zmq.asyncio.Context()
        self.push = self.ctx.socket(zmq.PUSH)
        self.push.bind(self.req_addr)
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.bind(self.out_addr)
        mpctx = mp.get_context("spawn")
        parent, child = mpctx.Pipe()
        self.process = mpctx.Process(target=engine_process_main, name="seventy-engine",
                                     args=(asdict(self.config), self.req_addr, self.out_addr, child))
        self.process.start()
        child.close()
        loop = asyncio.get_running_loop()
        kind, payload = await loop.run_in_executor(None, parent.recv)
        if kind != "ready":
            raise RuntimeError(f"the engine failed to start:\n{payload}")
        self.info = payload
        self.reader = asyncio.create_task(self._read())
        self.watch = asyncio.create_task(self._watch())

    async def _read(self) -> None:
        while True:
            msg = pickle.loads(await self.pull.recv())
            if msg[0] == "out":
                self.stats = msg[2]
                for out in msg[1]:
                    q = self.queues.get(out.id)
                    if q is not None:
                        q.put_nowait(out)
            elif msg[0] == "stats":
                self.stats = msg[1]
            elif msg[0] == "error":
                q = self.queues.get(msg[1])
                if q is not None:
                    q.put_nowait(ValueError(msg[2]))

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.process is not None and not self.process.is_alive():
                self.dead = f"the engine process exited with code {self.process.exitcode}"
                for q in self.queues.values():
                    q.put_nowait(RuntimeError(self.dead))
                os._exit(1)

    async def generate(self, prompt: list[int], params: SamplingParams, request_id: str | None = None, images=None,
                       arrival: float | None = None):
        """Yields RequestOutput objects (new tokens per step) until the request finishes. images: the
        seventy.multimodal.PromptImages of an expanded prompt; arrival: when the HTTP request came in (time.time),
        for the time-to-first-token and latency histograms."""
        if self.dead:
            raise RuntimeError(self.dead)
        arrival = arrival or time.time()
        first = True
        rid = request_id or uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self.queues[rid] = q
        await self.push.send(pickle.dumps(("add", rid, list(prompt), asdict(params), images)))
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                if first and item.new_tokens:
                    first = False
                    self.ttft.observe(time.time() - arrival)
                if item.finished:
                    self.latency.observe(time.time() - arrival)
                yield item
                if item.finished:
                    return
        finally:
            self.queues.pop(rid, None)

    async def abort(self, request_id: str) -> None:
        await self.push.send(pickle.dumps(("abort", request_id)))

    async def shutdown(self) -> None:
        if self.process is None:
            return
        await self.push.send(pickle.dumps(("shutdown",)))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.process.join, 120)
        if self.process.is_alive():
            self.process.terminate()
