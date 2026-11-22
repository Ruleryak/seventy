"""The batch sizes that get a CUDA graph of a speculative step: every size up to the limit by default, the powers of
two and the limit with SEVENTY_SPEC_BUCKETS=pow2."""

from seventy.engine.graphs import spec_buckets


def test_every_batch_size_gets_a_speculative_graph(monkeypatch):
    monkeypatch.delenv("SEVENTY_SPEC_BUCKETS", raising=False)
    assert spec_buckets(8) == (1, 2, 3, 4, 5, 6, 7, 8)
    assert spec_buckets(1) == (1,)


def test_powers_of_two_on_request(monkeypatch):
    assert spec_buckets(8, "pow2") == (1, 2, 4, 8)
    assert spec_buckets(12, "pow2") == (1, 2, 4, 8, 12)
    monkeypatch.setenv("SEVENTY_SPEC_BUCKETS", "pow2")
    assert spec_buckets(6) == (1, 2, 4, 6)
