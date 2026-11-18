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


def test_request_image_rows_rope_and_cache_key():
    import numpy as np

    from seventy import multimodal as mm

    pad = 248056
    grids = [(1, 4, 6)]
    img = mm.ImageInput(np.zeros((64, 96, 3), np.uint8), grids[0], "ab" * 32)
    ids, info = mm.expand_prompt([1, 2, pad, 3, 4], [img], pad, 2)
    r = Request("r", ids, SamplingParams(), images=info, cache_key=info.cache_key(ids))
    assert info.spans == [(2, 6)] and len(ids) == 10
    assert r.image_rows(0, 4) == [(2, 0, 0, 2)]
    assert r.image_rows(4, 6) == [(0, 0, 2, 4)]
    assert r.image_rows(8, 2) == []
    assert r.rope_delta == info.delta == int(info.positions.max()) + 1 - 10
    assert r.rope(0, 3) == [list(map(int, a)) for a in info.positions[:, :3]]
    r.output = [7, 8]
    assert r.rope(10, 2) == info.delta
    tail = r.rope(9, 2)
    assert tail[0] == [int(info.positions[0, 9]), 10 + info.delta] and tail[0] == tail[1] == tail[2]
    assert r.key_tokens[:2] == [1, 2] and r.key_tokens[2] < 0 and r.key_tokens[-2:] == [7, 8]
    plain = Request("p", [1, 2, 3], SamplingParams())
    assert plain.rope(0, 3) == 0 and plain.image_rows(0, 3) == [] and plain.key_tokens == [1, 2, 3]


def run_to_end(s, r):
    while r.status is not Status.FINISHED:
        run_step(s, s.schedule())


def test_prompt_snapshot_lets_the_same_prompt_compute_one_token():
    s = Scheduler(num_blocks=400, block_size=16, num_slots=6, max_model_len=8192, prefill_budget=4096, cache_entries=4)
    prompt = list(range(1000, 1900))
    a = Request("a", prompt, SamplingParams(max_tokens=1))
    s.add(a)
    plan = s.schedule()
    assert [w.count for w in plan.prefill] == [899]
    run_step(s, plan)
    plan = s.schedule()
    assert plan.copies and plan.copies[0][0] == "slot"
    assert [w.count for w in plan.decode] == [1]
    run_step(s, plan)
    assert a.status is Status.FINISHED
    assert sorted((len(e.tokens), e.snapshot) for e in s.cache.entries) == [(899, True), (900, False)]
    b = Request("b", list(prompt), SamplingParams(max_tokens=1))
    s.add(b)
    plan = s.schedule()
    assert b.computed == 899 and s.cache.hit_tokens == 899
    assert [w.count for w in plan.decode + plan.prefill] == [1]
    run_step(s, plan)
    run_to_end(s, b)
    assert len(s.cache.entries) == 2
    short = Request("c", list(range(100)), SamplingParams(max_tokens=1))
    s.add(short)
    run_to_end(s, short)
    assert not any(e.snapshot and len(e.tokens) == 99 for e in s.cache.entries)


def test_eviction_keeps_the_newest_turn_of_each_conversation():
    s = Scheduler(num_blocks=2000, block_size=16, num_slots=8, max_model_len=8192, prefill_budget=4096, cache_entries=3)
    conv = {}
    for name, base in (("x", 10_000), ("y", 20_000), ("z", 30_000)):
        r = Request(name, list(range(base, base + 600)), SamplingParams(max_tokens=3))
        s.add(r)
        run_to_end(s, r)
        conv[name] = r.tokens[: r.computed]
    finals = [e for e in s.cache.entries if not e.snapshot]
    assert len(s.cache.entries) == 3 and len(finals) >= 2
    nxt = Request("x2", conv["x"] + [7, 7, 5, 5], SamplingParams(max_tokens=3))
    s.add(nxt)
    run_step(s, s.schedule())
    assert nxt.computed > 600
    run_to_end(s, nxt)
    old_x = [e for e in s.cache.entries if e.tokens == conv["x"]]
    assert not old_x or old_x[0].superseded
    assert any(len(e.tokens) > len(conv["x"]) for e in s.cache.entries)
    free_blocks = len(s.blocks.free)
    s.cache.clear()
    assert len(s.slots.free) == 8 and len(s.blocks.free) == 2000 and free_blocks < 2000
