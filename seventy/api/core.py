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
    from ..engine.engine import Engine

    try:
        engine = Engine(EngineConfig(**config))
    except Exception:
        ready.send(("error", traceback.format_exc()))
        raise
    ready.send(("ready", {"kv_tokens": engine.kv_capacity_tokens, "pid": os.getpid(), "vision": engine.vision}))
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
    try:
        while alive:
            if not engine.has_work():
                if pull.poll(1000) == 0:
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
    finally:
        engine.close()


class EngineClient:
    """Starts the engine process and streams outputs of requests to the API process's coroutines."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.queues: dict[str, asyncio.Queue] = {}
        self.stats: dict = {}
        self.info: dict = {}
        self.process = None
        self.dead: str | None = None

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

    async def generate(self, prompt: list[int], params: SamplingParams, request_id: str | None = None, images=None):
        """Yields RequestOutput objects (new tokens per step) until the request finishes. images: the
        seventy.multimodal.PromptImages of an expanded prompt."""
        if self.dead:
            raise RuntimeError(self.dead)
        rid = request_id or uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self.queues[rid] = q
        await self.push.send(pickle.dumps(("add", rid, list(prompt), asdict(params), images)))
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
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
