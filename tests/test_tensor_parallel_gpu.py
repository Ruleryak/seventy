"""Tensor parallel on 2 GPUs, first 4 layers, FP16. Without peer-to-peer access between the GPUs, the engine does
every all-reduce with NCCL. The test fakes that case (torch reports no peer access) on any 2 GPUs and compares the
logits with one GPU, and with the custom all-reduce where the GPUs have peer access."""

import pytest
import torch
import torch.multiprocessing as mp

from seventy.ops import kernels_available

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() >= 2 and kernels_available()),
                       reason="needs 2 CUDA GPUs and the kernel library"),
]

PROMPT = [9707, 11, 1052, 374, 264, 1273, 315, 279, 5662, 13, 576, 3974, 13876, 38835, 34208, 916]


def run_rank(rank, size, port, path, peer, queue):
    """One rank: load the first 4 layers, prefill the prompt in two steps, and send rank 0's logits (FP32) and
    whether the shard uses the custom all-reduce."""
    if not peer:
        torch.cuda.can_device_access_peer = lambda a, b: False
    from seventy.engine import distributed
    from seventy.engine.cache import ModelCache
    from seventy.engine.worker import build_step, load_shard
    from seventy.parallel import Parallel

    distributed.share_cpus(size)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    if size > 1:
        distributed.init(rank, size, port, "nccl")
    cfg, model = load_shard(path, rank, size, dev, torch.float16, layers=4)
    cache = ModelCache(cfg.text, 4, 16, 1, torch.float16, dev, par=Parallel(rank, size))
    first, _ = model.forward(build_step([(PROMPT[:12], 0, [2, 0], 0, "all", 0, ())], 16, dev), cache)
    second, _ = model.forward(build_step([(PROMPT[12:], 12, [2, 0], 0, "all", 0, ())], 16, dev), cache)
    torch.cuda.synchronize()
    if rank == 0:
        queue.put((model.par.custom is not None, torch.cat((first, second)).float().cpu().numpy()))
    distributed.shutdown()


def logits_with(size, path, peer):
    """(whether rank 0 used the custom all-reduce, its logits) for size ranks; peer False fakes GPUs without peer
    access."""
    from seventy.engine import distributed

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=run_rank, args=(r, size, port, path, peer, queue)) for r in range(size)]
    for p in procs:
        p.start()
    custom, logits = queue.get(timeout=600)
    for p in procs:
        p.join(timeout=120)
    return custom, torch.from_numpy(logits)


def close(a, b):
    """Logits within 0.5% of the largest logit (the sums run in a different order; 4 layers on 2 V100 GPUs differ
    by about 0.05%)."""
    return (a - b).abs().max().item() < 0.005 * a.abs().max().item()


def test_nccl_all_reduce_without_peer_access(model_path):
    from seventy.engine.distributed import peer_access

    _, single = logits_with(1, str(model_path), True)
    custom, nccl = logits_with(2, str(model_path), False)
    assert not custom
    assert close(single, nccl), (single - nccl).abs().max().item()
    if peer_access(2):
        custom, fast = logits_with(2, str(model_path), True)
        assert custom
        assert close(fast, nccl), (fast - nccl).abs().max().item()
