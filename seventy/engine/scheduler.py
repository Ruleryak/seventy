"""Requests, cache allocation and the scheduling policy.

Each engine iteration runs up to two model steps: one decode step with one new token for every running sequence,
and one prefill step with prompt chunks of waiting sequences, up to a token budget. A sequence holds KV cache
blocks for all of its tokens and one Gated DeltaNet state slot. When blocks run out, the newest running sequence
is preempted: its blocks and slot are freed and it starts again from its prompt and the tokens it has generated.
"""

from __future__ import annotations

import enum
import itertools
from collections import deque
from dataclasses import dataclass, field

import torch

from .sampling import SamplingParams, TokenLogprobs


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    id: str
    prompt: list[int]
    params: SamplingParams
    arrival: int = 0
    output: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    computed: int = 0
    blocks: list[int] = field(default_factory=list)
    slot: int | None = None
    finish_reason: str | None = None
    generator: torch.Generator | None = None
    logprobs: list[TokenLogprobs] = field(default_factory=list)
    prompt_logprobs: list[TokenLogprobs | None] = field(default_factory=list)
    preemptions: int = 0

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def length(self) -> int:
        return len(self.prompt) + len(self.output)


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.free = deque(range(num_blocks))
        self.total = num_blocks

    def take(self, n: int) -> list[int] | None:
        if n > len(self.free):
            return None
        return [self.free.popleft() for _ in range(n)]

    def give(self, blocks: list[int]) -> None:
        self.free.extend(blocks)


class SlotAllocator:
    def __init__(self, num_slots: int):
        self.free = deque(range(num_slots))
        self.total = num_slots

    def take(self) -> int | None:
        return self.free.popleft() if self.free else None

    def give(self, slot: int) -> None:
        self.free.append(slot)


@dataclass
class Work:
    """One sequence's part of a model step: tokens [start, start + count) of request.tokens."""

    request: Request
    start: int
    count: int
    logits: str


@dataclass
class Plan:
    decode: list[Work] = field(default_factory=list)
    prefill: list[Work] = field(default_factory=list)
    finished: list[Request] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.decode and not self.prefill


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, num_slots: int, max_model_len: int,
                 prefill_budget: int = 4096, max_running: int | None = None):
        self.blocks = BlockAllocator(num_blocks)
        self.slots = SlotAllocator(num_slots)
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.prefill_budget = prefill_budget
        self.max_running = max_running or num_slots
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._arrivals = itertools.count()

    def add(self, request: Request) -> None:
        if len(request.prompt) == 0:
            raise ValueError("the prompt is empty")
        if len(request.prompt) >= self.max_model_len:
            raise ValueError(f"the prompt has {len(request.prompt)} tokens; the limit is {self.max_model_len - 1}")
        need = -(-min(self.max_model_len, len(request.prompt) + 1) // self.block_size)
        if need > self.blocks.total:
            raise ValueError("the prompt does not fit in the KV cache")
        request.arrival = next(self._arrivals)
        self.waiting.append(request)

    def abort(self, request_id: str) -> Request | None:
        for queue in (self.waiting, self.running):
            for r in list(queue):
                if r.id == request_id:
                    self._release(r)
                    queue.remove(r)
                    r.status, r.finish_reason = Status.FINISHED, "abort"
                    return r
        return None

    def _release(self, r: Request) -> None:
        self.blocks.give(r.blocks)
        r.blocks = []
        if r.slot is not None:
            self.slots.give(r.slot)
            r.slot = None

    def _grow(self, r: Request, tokens: int) -> bool:
        need = -(-tokens // self.block_size) - len(r.blocks)
        if need <= 0:
            return True
        got = self.blocks.take(need)
        if got is None:
            return False
        r.blocks += got
        return True

    def _preempt(self, r: Request) -> None:
        self._release(r)
        self.running.remove(r)
        r.computed = 0
        r.status = Status.WAITING
        r.preemptions += 1
        self.waiting.appendleft(r)

    def schedule(self) -> Plan:
        plan = Plan()
        decoders = [r for r in self.running if r.computed == r.length - 1]
        for r in sorted(decoders, key=lambda x: x.arrival):
            if r.status is not Status.RUNNING:
                continue
            while not self._grow(r, r.length):
                victim = max((x for x in self.running if x is not r), key=lambda x: x.arrival, default=None)
                if victim is None or victim.arrival < r.arrival:
                    victim = r
                self._preempt(victim)
                if victim is r:
                    break
            if r.status is Status.RUNNING:
                plan.decode.append(Work(r, r.computed, 1, "last"))
        budget = self.prefill_budget
        for r in [x for x in self.running if x.computed < x.length - 1]:
            budget = self._plan_prefill(plan, r, budget)
        while self.waiting and budget > 0 and len(self.running) < self.max_running:
            r = self.waiting[0]
            slot = self.slots.take()
            if slot is None:
                break
            r.slot = slot
            if not self._grow(r, min(r.length, r.computed + budget)):
                self.slots.give(slot)
                r.slot = None
                break
            self.waiting.popleft()
            r.status = Status.RUNNING
            self.running.append(r)
            budget = self._plan_prefill(plan, r, budget)
        return plan

    def _plan_prefill(self, plan: Plan, r: Request, budget: int) -> int:
        remaining = r.length - r.computed
        count = min(remaining, budget)
        if count <= 0 or not self._grow(r, r.computed + count):
            return budget
        last = r.computed + count == r.length
        want = "all" if r.params.prompt_logprobs is not None else ("last" if last else "none")
        plan.prefill.append(Work(r, r.computed, count, want))
        return budget - count

    def finish(self, r: Request, reason: str) -> None:
        self._release(r)
        if r in self.running:
            self.running.remove(r)
        r.status, r.finish_reason = Status.FINISHED, reason

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
