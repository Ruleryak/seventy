from seventy.engine.sampling import SamplingParams
from seventy.engine.scheduler import Request, Scheduler, Status


def run_step(s: Scheduler, plan, next_token=7):
    for w in plan.decode + plan.prefill:
        r = w.request
        r.computed = w.start + w.count
        if r.computed == r.length:
            r.output.append(next_token)
            if len(r.output) >= r.params.max_tokens:
                s.finish(r, "length")


def test_prefill_is_chunked_then_decodes():
    s = Scheduler(num_blocks=64, block_size=16, num_slots=4, max_model_len=4096, prefill_budget=100)
    r = Request("a", list(range(250)), SamplingParams(max_tokens=3))
    s.add(r)
    counts = []
    for _ in range(3):
        plan = s.schedule()
        counts.append([w.count for w in plan.prefill])
        run_step(s, plan)
    assert counts == [[100], [100], [50]]
    assert r.output == [7]
    plan = s.schedule()
    assert [w.count for w in plan.decode] == [1] and not plan.prefill
    run_step(s, plan)
    run_step(s, s.schedule())
    assert r.status is Status.FINISHED and r.finish_reason == "length"
    assert len(s.blocks.free) == 64 and len(s.slots.free) == 4


def test_decode_batch_and_new_prefill_in_one_iteration():
    s = Scheduler(num_blocks=64, block_size=16, num_slots=4, max_model_len=4096, prefill_budget=64)
    a = Request("a", list(range(10)), SamplingParams(max_tokens=10))
    s.add(a)
    run_step(s, s.schedule())
    b = Request("b", list(range(30)), SamplingParams(max_tokens=10))
    s.add(b)
    plan = s.schedule()
    assert [w.request.id for w in plan.decode] == ["a"]
    assert [(w.request.id, w.count) for w in plan.prefill] == [("b", 30)]


def test_preemption_frees_newest_and_recomputes():
    s = Scheduler(num_blocks=6, block_size=16, num_slots=4, max_model_len=4096, prefill_budget=4096)
    a = Request("a", list(range(40)), SamplingParams(max_tokens=100))
    b = Request("b", list(range(40)), SamplingParams(max_tokens=100))
    s.add(a)
    s.add(b)
    run_step(s, s.schedule())
    assert len(a.blocks) == 3 and len(b.blocks) == 3 and not s.blocks.free
    for _ in range(10):
        run_step(s, s.schedule())
    assert b.preemptions >= 1 and b.status is Status.WAITING and b.computed == 0
    assert a.status is Status.RUNNING
    assert len(a.blocks) == -(-a.length // 16)


def test_abort_releases():
    s = Scheduler(num_blocks=8, block_size=16, num_slots=2, max_model_len=4096)
    r = Request("x", list(range(20)), SamplingParams())
    s.add(r)
    s.schedule()
    s.abort("x")
    assert r.finish_reason == "abort" and len(s.blocks.free) == 8 and len(s.slots.free) == 2


def finish_all(s, steps=50):
    for _ in range(steps):
        if not s.has_work():
            return
        run_step(s, s.schedule())


def test_prefix_cache_reuses_finished_sequence():
    s = Scheduler(num_blocks=64, block_size=16, num_slots=4, max_model_len=4096, max_running=2, cache_entries=2)
    a = Request("a", list(range(40)), SamplingParams(max_tokens=3))
    s.add(a)
    finish_all(s)
    assert len(s.cache.entries) == 1 and s.cache.entries[0].tokens == a.tokens[: a.computed]
    turn2 = a.tokens + [900, 901, 902]
    b = Request("b", turn2, SamplingParams(max_tokens=2))
    s.add(b)
    plan = s.schedule()
    assert b.computed == len(s.cache.entries[0].tokens)
    n = len(s.cache.entries[0].tokens)
    assert [w.count for w in plan.prefill] == [len(turn2) - n]
    kinds = sorted(c[0] for c in plan.copies)
    assert kinds == (["block", "slot"] if n % 16 else ["slot"])
    shared = s.cache.entries[0].blocks[: n // 16]
    assert b.blocks[: n // 16] == shared and all(s.blocks.refs[x] == 2 for x in shared)


def test_prefix_cache_evicts_before_preempting_and_frees_everything():
    s = Scheduler(num_blocks=8, block_size=16, num_slots=3, max_model_len=4096, max_running=2, cache_entries=1)
    a = Request("a", list(range(60)), SamplingParams(max_tokens=2))
    s.add(a)
    finish_all(s)
    assert len(s.cache.entries) == 1
    b = Request("b", [5] * 100, SamplingParams(max_tokens=2))
    s.add(b)
    finish_all(s)
    assert b.status is Status.FINISHED and b.preemptions == 0
    s.cache.clear()
    assert len(s.blocks.free) == 8 and len(s.slots.free) == 3 and all(x == 0 for x in s.blocks.refs)


def test_no_match_no_reuse():
    s = Scheduler(num_blocks=32, block_size=16, num_slots=3, max_model_len=4096, max_running=2, cache_entries=1)
    s.add(Request("a", list(range(30)), SamplingParams(max_tokens=1)))
    finish_all(s)
    b = Request("b", [7] + list(range(30)), SamplingParams(max_tokens=1))
    s.add(b)
    plan = s.schedule()
    assert not plan.copies and plan.prefill[0].start == 0
