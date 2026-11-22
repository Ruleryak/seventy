"""Requests, cache allocation and the scheduling policy.

Each engine iteration runs up to two model steps: one decode step with one new token for every running sequence,
and one prefill step with prompt chunks of waiting sequences, up to a token budget. A sequence holds KV cache
blocks for all of its tokens and one Gated DeltaNet state slot. When blocks run out, the least recently used
prefix-cache entries are evicted first; then the newest running sequence is preempted: its blocks and slot are
freed and it starts again from its prompt and the tokens it has generated.

Prefix cache: when a sequence finishes, its KV blocks and state slot (the state after its last computed token)
become a cache entry instead of being freed. A new request whose tokens start with an entry's tokens shares the
entry's full KV blocks (reference counted), gets a copy of the entry's partly filled last block and of its state
slot, and computes only the rest. The copies are part of the plan; the runner makes them before the step. Entries
are matched on key_tokens, where image tokens stand for the image's content (seventy.multimodal).

Prompt snapshots: for prompts of SNAPSHOT_MIN tokens or more, prefill stops one token before the end of the prompt;
the state at that point becomes an entry too, so the same prompt sent again computes only its last token (and a
request that diverges inside its output still reuses its prompt).

Eviction takes spare entries first, then the least recently used. An entry is spare when a newer entry extends it (a
conversation's earlier turn; a snapshot only when a newer snapshot extends it: a final entry never supersedes the
snapshot of its own prompt, which the next turn needs when it diverges inside the output), or when it diverged: a
request matched a shorter entry with the same tokens at the start and went on differently (agent clients send the
next turn without the model's earlier thinking, so the final entries of their turns never match again).
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
    draft: list[int] = field(default_factory=list)
    draft_q: torch.Tensor | None = None
    draft_cand: torch.Tensor | None = None
    cacheable: bool = True
    images: object | None = None
    cache_key: list[int] | None = None
    snapshot_done: bool = False
    blocks_epoch: int = 0

    def __setattr__(self, name, value):
        """A new blocks list (not an append to the current one) bumps blocks_epoch, so that the spec graphs know
        when a sequence's block table must be sent whole."""
        if name == "blocks" and value is not self.__dict__.get("blocks"):
            object.__setattr__(self, "blocks_epoch", self.__dict__.get("blocks_epoch", 0) + 1)
        object.__setattr__(self, name, value)

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output

    @property
    def last_token(self) -> int:
        """tokens[-1], without building the token list (decode steps call this for every sequence)."""
        return self.output[-1] if self.output else self.prompt[-1]

    def token_range(self, start: int, end: int) -> list[int]:
        """tokens[start:end], without building the whole token list."""
        p = len(self.prompt)
        if end <= p:
            return self.prompt[start:end]
        if start >= p:
            return self.output[start - p : end - p]
        return self.prompt[start:] + self.output[: end - p]

    @property
    def key_tokens(self) -> list[int]:
        return self.tokens if self.cache_key is None else self.cache_key + self.output

    @property
    def rope_delta(self) -> int:
        """Rotary position minus token index for every token after the last image (0 without images)."""
        return self.images.delta if self.images is not None else 0

    def rope(self, start: int, count: int):
        """Rotary positions of tokens [start, start + count): an int delta when they are index + delta on all
        three axes, else explicit [3][count] lists."""
        if self.images is None:
            return 0
        n = len(self.prompt)
        delta = self.images.delta
        if start >= n:
            return delta
        end = start + count
        pos = [list(map(int, axis)) for axis in self.images.positions[:, start : min(end, n)]]
        if end > n:
            for axis in pos:
                axis += range(n + delta, end + delta)
        return pos

    def image_rows(self, start: int, count: int) -> list[tuple[int, int, int, int]]:
        """(row in the chunk, image index, first image token, tokens) of image tokens in [start, start + count)."""
        if self.images is None:
            return []
        out = []
        for i, (off, n) in enumerate(self.images.spans):
            lo, hi = max(start, off), min(start + count, off + n)
            if lo < hi:
                out.append((lo - start, i, lo - off, hi - lo))
        return out

    @property
    def length(self) -> int:
        return len(self.prompt) + len(self.output)


class BlockAllocator:
    """KV blocks with reference counts: a block returns to the free list when its last user gives it back."""

    def __init__(self, num_blocks: int):
        self.free = deque(range(num_blocks))
        self.total = num_blocks
        self.refs = [0] * num_blocks

    def take(self, n: int) -> list[int] | None:
        if n > len(self.free):
            return None
        out = [self.free.popleft() for _ in range(n)]
        for b in out:
            self.refs[b] = 1
        return out

    def share(self, blocks: list[int]) -> None:
        for b in blocks:
            self.refs[b] += 1

    def give(self, blocks: list[int]) -> None:
        for b in blocks:
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self.free.append(b)


class SlotAllocator:
    def __init__(self, num_slots: int):
        self.free = deque(range(num_slots))
        self.total = num_slots

    def take(self) -> int | None:
        return self.free.popleft() if self.free else None

    def give(self, slot: int) -> None:
        self.free.append(slot)


SNAPSHOT_MIN = 512
PROMPT_LOGPROBS_CHUNK = 512
"""Most prompt tokens per step of all requests with prompt_logprobs together: every row of their chunks gets FP32
logits over the whole vocabulary (about 1 MB per row for Qwen3.5)."""


@dataclass
class CacheEntry:
    tokens: list[int]
    blocks: list[int]
    slot: int
    used: int
    snapshot: bool = False
    superseded: bool = False
    diverged: bool = False

    @property
    def spare(self) -> bool:
        return self.superseded or self.diverged


class PrefixCache:
    def __init__(self, blocks: BlockAllocator, slots: SlotAllocator, max_entries: int):
        self.blocks = blocks
        self.slots = slots
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self.clock = itertools.count()
        self.pinned: CacheEntry | None = None
        self.hits = 0
        self.hit_tokens = 0

    def lookup(self, tokens: list[int]) -> CacheEntry | None:
        """The entry with the longest token prefix of `tokens` (leaving at least one token to compute)."""
        best = None
        for e in self.entries:
            n = len(e.tokens)
            if n < len(tokens) and (best is None or n > len(best.tokens)) and tokens[:n] == e.tokens:
                best = e
        if best is not None:
            best.used = next(self.clock)
            n = len(best.tokens)
            for e in self.entries:
                if len(e.tokens) > n and not e.diverged and e.tokens[:n] == best.tokens:
                    e.diverged = True
        return best

    def find(self, tokens: list[int]) -> CacheEntry | None:
        """The entry with exactly these tokens."""
        n = len(tokens)
        return next((e for e in self.entries if len(e.tokens) == n and e.tokens == tokens), None)

    def add(self, tokens: list[int], blocks: list[int], slot: int, snapshot: bool = False) -> None:
        if not self.max_entries:
            self.blocks.give(blocks)
            self.slots.give(slot)
            return
        n = len(tokens)
        for e in self.entries:
            if len(e.tokens) == n and e.tokens == tokens:
                e.used = next(self.clock)
                e.snapshot = e.snapshot and snapshot
                self.blocks.give(blocks)
                self.slots.give(slot)
                return
        for e in self.entries:
            if (not e.snapshot or snapshot) and len(e.tokens) < n and tokens[: len(e.tokens)] == e.tokens:
                e.superseded = True
        while len(self.entries) >= self.max_entries:
            self.evict()
        self.entries.append(CacheEntry(tokens, blocks, slot, next(self.clock), snapshot))

    def evict(self) -> bool:
        if not self.entries:
            return False
        candidates = [x for x in self.entries if x is not self.pinned]
        if not candidates:
            return False
        e = min(candidates, key=lambda x: (not x.spare, x.used))
        self.entries.remove(e)
        self.blocks.give(e.blocks)
        self.slots.give(e.slot)
        return True

    def clear(self) -> None:
        while self.evict():
            pass


@dataclass
class Work:
    """One sequence's part of a model step: tokens [start, start + count) of request.tokens, or `tokens` (a
    speculative verify: the newest token and the drafts)."""

    request: Request
    start: int
    count: int
    logits: str
    tokens: list[int] | None = None

    def ids(self) -> list[int]:
        return self.tokens if self.tokens is not None else self.request.token_range(self.start, self.start + self.count)


@dataclass
class Plan:
    decode: list[Work] = field(default_factory=list)
    prefill: list[Work] = field(default_factory=list)
    finished: list[Request] = field(default_factory=list)
    copies: list[tuple[str, int, int]] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.decode and not self.prefill


class Scheduler:
    def __init__(self, num_blocks: int, block_size: int, num_slots: int, max_model_len: int,
                 prefill_budget: int = 4096, max_running: int | None = None, cache_entries: int = 0):
        self.blocks = BlockAllocator(num_blocks)
        self.slots = SlotAllocator(num_slots)
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.prefill_budget = prefill_budget
        self.max_running = max_running or num_slots
        self.cache = PrefixCache(self.blocks, self.slots, cache_entries)
        self.preempted = 0
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
        for r in list(self.running):
            if r.id == request_id:
                self.finish(r, "abort")
                return r
        for r in list(self.waiting):
            if r.id == request_id:
                self._release(r)
                self.waiting.remove(r)
                r.status, r.finish_reason = Status.FINISHED, "abort"
                return r
        return None

    def _release(self, r: Request) -> None:
        self.blocks.give(r.blocks)
        r.blocks = []
        if r.slot is not None:
            self.slots.give(r.slot)
            r.slot = None

    def _take_blocks(self, n: int) -> list[int] | None:
        got = self.blocks.take(n)
        while got is None and self.cache.evict():
            got = self.blocks.take(n)
        return got

    def _take_slot(self) -> int | None:
        slot = self.slots.take()
        while slot is None and self.cache.evict():
            slot = self.slots.take()
        return slot

    def _grow(self, r: Request, tokens: int) -> bool:
        need = -(-tokens // self.block_size) - len(r.blocks)
        if need <= 0:
            return True
        got = self._take_blocks(need)
        if got is None:
            return False
        r.blocks += got
        return True

    def _reuse(self, r: Request, entry: CacheEntry | None) -> list[tuple[str, int, int]]:
        """Start a fresh request from a matching prefix-cache entry (found before its slot was taken, and pinned
        meanwhile). Returns the copies to make."""
        if entry is None or entry not in self.cache.entries:
            return []
        n = len(entry.tokens)
        full = n // self.block_size
        copies = []
        self.blocks.share(entry.blocks[:full])
        r.blocks = list(entry.blocks[:full])
        if n % self.block_size:
            extra = self._take_blocks(1)
            if extra is None:
                self.blocks.give(r.blocks)
                r.blocks = []
                return []
            copies.append(("block", entry.blocks[full], extra[0]))
            r.blocks.append(extra[0])
        copies.append(("slot", entry.slot, r.slot))
        r.computed = n
        self.cache.hits += 1
        self.cache.hit_tokens += n
        return copies

    def _preempt(self, r: Request) -> None:
        self._release(r)
        r.draft, r.draft_q, r.draft_cand = [], None, None
        self.running.remove(r)
        r.computed = 0
        r.status = Status.WAITING
        r.preemptions += 1
        self.preempted += 1
        self.waiting.appendleft(r)

    def schedule(self) -> Plan:
        plan = Plan()
        decoders = [r for r in self.running if r.computed == r.length - 1]
        for r in sorted(decoders, key=lambda x: x.arrival):
            if r.status is not Status.RUNNING:
                continue
            if self._snapshot_point(r) == r.computed:
                self._take_snapshot(plan, r)
            if r.draft and r.computed + 1 + len(r.draft) > self.max_model_len:
                r.draft, r.draft_q, r.draft_cand = [], None, None
            count = 1 + len(r.draft)
            while not self._grow(r, r.computed + count):
                victim = max((x for x in self.running if x is not r), key=lambda x: x.arrival, default=None)
                if victim is None or victim.arrival < r.arrival:
                    victim = r
                self._preempt(victim)
                if victim is r:
                    break
            if r.status is Status.RUNNING:
                if r.draft:
                    plan.decode.append(Work(r, r.computed, count, "all", tokens=[r.last_token] + r.draft))
                else:
                    plan.decode.append(Work(r, r.computed, 1, "last"))
        budget = self.prefill_budget
        for r in [x for x in self.running if x.computed < x.length - 1]:
            budget = self._plan_prefill(plan, r, budget)
        while self.waiting and budget > 0 and len(self.running) < self.max_running:
            r = self.waiting[0]
            entry = self.cache.lookup(r.key_tokens) if r.computed == 0 and self.cache.max_entries else None
            self.cache.pinned = entry
            slot = self._take_slot()
            if slot is None:
                self.cache.pinned = None
                break
            r.slot = slot
            copies = self._reuse(r, entry)
            self.cache.pinned = None
            if not self._grow(r, min(r.length, r.computed + budget)):
                self._release(r)
                r.computed = 0
                break
            plan.copies += copies
            self.waiting.popleft()
            r.status = Status.RUNNING
            self.running.append(r)
            budget = self._plan_prefill(plan, r, budget)
        return plan

    def _snapshot_point(self, r: Request) -> int | None:
        """Where this request's prompt snapshot goes (one token before the end of the prompt), if it gets one."""
        if not self.cache.max_entries or r.snapshot_done or not r.cacheable or len(r.prompt) < SNAPSHOT_MIN:
            return None
        return len(r.prompt) - 1

    def _take_snapshot(self, plan: Plan, r: Request) -> None:
        r.snapshot_done = True
        key = r.key_tokens[: r.computed]
        same = self.cache.find(key)
        if same is not None:
            same.used = next(self.cache.clock)
            return
        slot = self.slots.take()
        while slot is None and self.cache.entries and self.cache.evict():
            slot = self.slots.take()
        if slot is None:
            return
        n = r.computed
        keep = -(-n // self.block_size)
        self.blocks.share(r.blocks[:keep])
        plan.copies.append(("slot", r.slot, slot))
        self.cache.add(key, list(r.blocks[:keep]), slot, snapshot=True)

    def _plan_prefill(self, plan: Plan, r: Request, budget: int) -> int:
        point = self._snapshot_point(r)
        if point is not None and r.computed == point:
            self._take_snapshot(plan, r)
        remaining = r.length - r.computed
        if point is not None and r.computed < point:
            remaining = point - r.computed
        count = min(remaining, budget)
        if r.params.prompt_logprobs is not None:
            count = min(count, PROMPT_LOGPROBS_CHUNK - sum(w.count for w in plan.prefill if w.logits == "all"))
        if count <= 0 or not self._grow(r, r.computed + count):
            return budget
        last = r.computed + count == r.length
        want = "all" if r.params.prompt_logprobs is not None else ("last" if last else "none")
        plan.prefill.append(Work(r, r.computed, count, want))
        return budget - count

    def finish(self, r: Request, reason: str) -> None:
        if self.cache.max_entries and r.slot is not None and r.computed > 0 and r.cacheable:
            keep = -(-r.computed // self.block_size)
            self.blocks.give(r.blocks[keep:])
            self.cache.add(r.key_tokens[: r.computed], r.blocks[:keep], r.slot)
            r.blocks, r.slot = [], None
        else:
            self._release(r)
        if r in self.running:
            self.running.remove(r)
        r.status, r.finish_reason = Status.FINISHED, reason

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
