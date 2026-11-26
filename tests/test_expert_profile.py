"""Expert profiles (seventy/model/moe.py): the counts of the ranks of a run and of later runs add up in one file,
the ranking puts the most used experts first, a file for another model is not used, and the built-in profile of
Qwen3.8-Flash-Next matches its shape."""

from types import SimpleNamespace

import torch

from seventy.model import moe


def _cfg(layers=2, experts=8):
    return SimpleNamespace(exp=object(), num_layers=layers, num_experts=experts, experts_per_token=2, hidden_size=64)


def _layer(prefix, e0, uses):
    m = moe.MoE.__new__(moe.MoE)
    m.prefix, m.e0 = prefix, e0
    m.cache = SimpleNamespace(uses=torch.tensor(uses, dtype=torch.int32))
    return m


def test_counts_of_ranks_and_runs_add_up(tmp_path):
    path = tmp_path / "profile.json"
    cfg = _cfg()
    moe.save_profile(path, cfg, [_layer("l.0.mlp", 0, [5, 0, 1, 9]), _layer("l.1.mlp", 0, [0, 0, 0, 1])])
    moe.save_profile(path, cfg, [_layer("l.0.mlp", 4, [2, 7, 0, 0]), _layer("l.1.mlp", 4, [3, 3, 0, 0])])
    moe.save_profile(path, cfg, [_layer("l.0.mlp", 0, [1, 0, 0, 0])])
    ranking = moe.load_profile(path, cfg)
    assert ranking["l.0.mlp"] == [3, 5, 0, 4, 2, 1, 6, 7]
    assert ranking["l.1.mlp"][:3] == [4, 5, 3]
    assert moe.load_profile(path, _cfg(experts=16)) is None
    moe.save_profile(path, _cfg(experts=16), [_layer("l.0.mlp", 0, [1] * 16)])
    assert moe.load_profile(path, cfg) is None and moe.load_profile(path, _cfg(experts=16)) is not None


def test_builtin_flash_next_profile():
    cfg = SimpleNamespace(exp=object(), num_layers=48, num_experts=512, experts_per_token=10, hidden_size=2560)
    ranking = moe.builtin_profile(cfg)
    assert ranking is not None and len(ranking) == 48
    assert all(sorted(r) == list(range(512)) for r in ranking.values())
    assert moe.builtin_profile(_cfg()) is None
