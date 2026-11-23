"""The shapes of the speculative CUDA graphs: the batch sizes that get a graph (every size up to the limit by
default, the powers of two and the limit with SEVENTY_SPEC_BUCKETS=pow2), the rows each graph verifies for each
sequence (the whole drafter block up to a budget of rows per step, then fewer), and the packed inputs of a graph
that verifies fewer rows than the drafter drafts."""

import types

import pytest
import torch

from seventy.engine.graphs import spec_buckets
from seventy.engine.spec_graphs import MIN_ROWS, ROW_BUDGET, SpecGraphs, verify_rows


def test_every_batch_size_gets_a_speculative_graph(monkeypatch):
    monkeypatch.delenv("SEVENTY_SPEC_BUCKETS", raising=False)
    assert spec_buckets(8) == (1, 2, 3, 4, 5, 6, 7, 8)
    assert spec_buckets(1) == (1,)


def test_powers_of_two_on_request(monkeypatch):
    assert spec_buckets(8, "pow2") == (1, 2, 4, 8)
    assert spec_buckets(12, "pow2") == (1, 2, 4, 8, 12)
    monkeypatch.setenv("SEVENTY_SPEC_BUCKETS", "pow2")
    assert spec_buckets(6) == (1, 2, 4, 6)


def test_verify_rows_keep_about_the_row_budget_per_step(monkeypatch):
    monkeypatch.delenv("SEVENTY_SPEC_ROW_BUDGET", raising=False)
    assert (ROW_BUDGET, MIN_ROWS) == (32, 4)
    assert [verify_rows(b, 8) for b in range(1, 9)] == [8, 8, 8, 8, 6, 5, 4, 4]
    assert verify_rows(16, 8) == MIN_ROWS
    assert verify_rows(8, 3) == 3
    assert [verify_rows(b, 8, 0) for b in (1, 8)] == [8, 8]
    monkeypatch.setenv("SEVENTY_SPEC_ROW_BUDGET", "0")
    assert verify_rows(8, 8) == 8


@pytest.mark.gpu
def test_a_graph_of_fewer_rows_takes_the_first_drafts():
    g = SpecGraphs.__new__(SpecGraphs)
    g.block, g.top, g.width, g.device = 8, 4, 3, torch.device("cuda")
    g.cache = types.SimpleNamespace(scratch_slot=99, scratch_block=7)
    g.buckets, g.rows, g.packed, g.sent = (2,), {2: 5}, {}, {}
    fields = {name: shape for name, shape, _, _ in g._fields(2)}
    assert fields["tokens"] == (2, 5) and fields["q"] == (2, 4, 4) and fields["u_residual"] == (2, 5)
    assert fields["gumbel"] == (2, 7, 4)
    g._inputs(2)
    row = {"tokens": list(range(10, 18)), "start": 40, "slot": 3, "table": [1, 2, 3], "temp": 0.5, "top_k": 4,
           "top_p": 1.0, "min_p": 0.0, "q": torch.arange(28.0).view(7, 4), "cand": torch.arange(28).view(7, 4),
           "u_accept": torch.arange(7.0), "u_residual": torch.arange(8.0), "gumbel": torch.ones(7, 4), "limit": 100}
    g.pack([row])
    views = g.packed[2][2]
    assert views["tokens"][0].tolist() == [10, 11, 12, 13, 14]
    assert views["q"][0].tolist() == torch.arange(16.0).view(4, 4).tolist()
    assert views["u_accept"][0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert views["u_residual"][0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert views["gumbel"][0].shape == (7, 4) and views["limit"][0] == 5
    assert views["tokens"][1].tolist() == [0] * 5 and views["slots"][1] == 99
