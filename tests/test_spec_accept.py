import torch

from seventy.engine.sampling import RowState, SamplingParams
from seventy.engine.spec import KMAX, accept, device_accept


def test_greedy_device_accept_matches_host():
    torch.manual_seed(0)
    b, k, v = 5, 7, 300
    logits = torch.randn(b, k + 1, v)
    pred = logits.argmax(-1)
    drafts = pred[:, :k].clone()
    drafts[1, 3] = (drafts[1, 3] + 1) % v
    drafts[2, 0] = (drafts[2, 0] + 1) % v
    q = torch.zeros(b, k, 16)
    q[..., 0] = 1
    cand = drafts[..., None].expand(b, k, 16).clone()
    z = torch.zeros(b)
    n, new = device_accept(logits, drafts, q, cand, z, torch.ones(b, dtype=torch.long), torch.ones(b), z,
                           torch.rand(b, k), torch.rand(b, k + 1))
    for i in range(b):
        row = RowState(SamplingParams(temperature=0), [1], [], None)
        hn, hnew = accept(logits[i], drafts[i].tolist(), None, None, row, ())
        assert int(n[i]) == hn and new[i, : hn + 1].tolist() == hnew


def test_sampled_device_accept_preserves_target_distribution():
    torch.manual_seed(1)
    v, k, trials = 12, 1, 40000
    logits = torch.randn(1, k + 1, v).expand(trials, -1, -1).contiguous()
    cand = torch.arange(16).remainder(v)[None, None].expand(trials, k, 16).clone()
    qraw = torch.softmax(torch.randn(16), 0)
    q = qraw[None, None].expand(trials, k, 16).clone()
    drafts = cand[:, 0].gather(1, torch.multinomial(qraw, trials, replacement=True)[:, None])
    t = torch.ones(trials)
    n, new = device_accept(logits, drafts, q, cand, t, torch.full((trials,), 5), torch.full((trials,), 0.9),
                           torch.zeros(trials), torch.rand(trials, k), torch.rand(trials, k + 1))
    first = new[:, 0]
    counts = torch.bincount(first, minlength=v).float() / trials
    vals, idx = torch.topk(logits[0, 0], 5)
    p = torch.softmax(vals, 0)
    keep = (torch.cumsum(p, 0) - p) < 0.9
    p = p * keep
    p = p / p.sum()
    want = torch.zeros(v).scatter(0, idx, p)
    assert (counts - want).abs().max() < 0.01, (counts, want)


def test_cut_at_stops_matches_host():
    from seventy.engine.spec import cut_at_stops, cut_host

    g = torch.Generator().manual_seed(5)
    for _ in range(200):
        b, k1 = 4, 8
        new = torch.randint(0, 6, (b, k1), generator=g)
        n = torch.randint(0, k1, (b,), generator=g)
        stop = torch.full((b, 3), -1, dtype=torch.long)
        stop[:, 0] = 5
        stop[1, 1] = 4
        limit = torch.randint(1, k1 + 2, (b,), generator=g)
        got = cut_at_stops(n, new, stop, limit)
        for i in range(b):
            want = cut_host(int(n[i]), new[i].tolist(), {t for t in stop[i].tolist() if t >= 0}, int(limit[i]))
            assert int(got[i]) == want
            assert int(got[i]) <= int(n[i]) and int(got[i]) + 1 <= max(1, int(limit[i]))


def test_merge_top_of_shards_matches_full_logits():
    from seventy.engine.spec import merge_top, top_of

    g = torch.Generator().manual_seed(9)
    logits = torch.randn(6, 9, 400, generator=g)
    logits[0, 0, 7] = logits[0, 0, 301] = 50.0
    shards = logits.split(100, dim=-1)
    vals, ids = [], []
    for s, part in enumerate(shards):
        v, i = torch.topk(part, 16, dim=-1)
        vals.append(v)
        ids.append(i + 100 * s)
    pred, v, i = merge_top(vals, ids, 16)
    want_pred, want_v, want_i = top_of(logits, 16)
    assert torch.equal(pred, want_pred) and pred[0, 0] == 7
    assert torch.equal(v, want_v)
    assert torch.equal(torch.gather(logits, -1, i), want_v)


def test_logprobs_of_in_row_blocks_matches_one_block():
    import torch

    from seventy.engine.sampling import logprobs_of

    torch.manual_seed(0)
    logits = torch.randn(300, 1000) * 3
    tokens = torch.randint(0, 1000, (300,))
    a = logprobs_of(logits, tokens, 3, rows=128)
    b = logprobs_of(logits, tokens, 3, rows=1000)
    assert len(a) == 300
    assert [(x.token, x.rank, [t for t, _ in x.top]) for x in a] == [(x.token, x.rank, [t for t, _ in x.top]) for x in b]
    assert max(abs(x.logprob - y.logprob) for x, y in zip(a, b)) < 1e-5
