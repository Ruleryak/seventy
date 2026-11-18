"""Tensor-parallel sharding on the CPU: 2 and 4 processes over gloo against one process, first 4 layers, FP32."""

import os

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = [pytest.mark.model, pytest.mark.slow]

PROMPT = [9707, 11, 1052, 374, 264, 1273, 315, 279, 5662, 13, 576, 3974, 13876, 38835, 34208, 916]


def run_rank(rank, size, port, path, queue):
    torch.set_num_threads(4)
    from seventy.engine import distributed
    from seventy.engine.cache import ModelCache
    from seventy.engine.worker import build_step, load_shard
    from seventy.parallel import Parallel

    if size > 1:
        distributed.init(rank, size, port, "gloo")
    cfg, model = load_shard(path, rank, size, "cpu", torch.float32, layers=4)
    cache = ModelCache(cfg.text, 4, 16, 1, torch.float32, "cpu", par=Parallel(rank, size))
    first, _ = model.forward(build_step([(PROMPT[:12], 0, [2, 0], 0, "all")], 16, "cpu"), cache)
    second, _ = model.forward(build_step([(PROMPT[12:], 12, [2, 0], 0, "all")], 16, "cpu"), cache)
    if rank == 0:
        queue.put(torch.cat((first, second)).numpy())
    distributed.shutdown()


def logits_with(size, path):
    from seventy.engine import distributed

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=run_rank, args=(r, size, port, path, queue)) for r in range(size)]
    for p in procs:
        p.start()
    out = torch.from_numpy(queue.get(timeout=600))
    for p in procs:
        p.join(timeout=120)
    return out


@pytest.mark.parametrize("size", [2, 4])
def test_sharded_matches_single(model_path, size):
    single = logits_with(1, str(model_path))
    sharded = logits_with(size, str(model_path))
    diff = (single - sharded).abs().max().item()
    assert diff < 1e-3 * max(single.abs().max().item(), 1.0), diff
