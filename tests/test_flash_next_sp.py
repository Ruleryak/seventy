"""The sequence-parallel prefill of Qwen4-Exp on 4 GPUs (Qwen4ExpTextModel._forward_sp: the streams split by rows
over the ranks) against the replicated path, on the first 4 layers of a real checkpoint (flash_next_path: Gated
DeltaNet layers, the PLE layer, a QSA attention layer, the experts; with the KV split of 4 GPUs). Sequence A
prefills 1500 tokens with the split, sequence B the same tokens without it; the last row's logits must agree, and
so must four decode steps after it (the caches, the Gated DeltaNet and PLE states of both prefills)."""

import pytest
import torch
import torch.multiprocessing as mp

from seventy.ops import kernels_available

pytestmark = [pytest.mark.gpu, pytest.mark.model,
              pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() >= 4
                                      and kernels_available()),
                                 reason="needs 4 CUDA GPUs and the kernel library")]


def run_rank(rank, port, path, queue):
    from seventy import memory
    from seventy.engine import distributed
    from seventy.engine.cache import ModelCache
    from seventy.engine.step import SequenceChunk, StepInput
    from seventy.engine.worker import load_shard
    from seventy.model import qwen4_exp
    from seventy.parallel import Parallel

    distributed.share_cpus(4)
    torch.cuda.set_device(rank)
    distributed.init(rank, 4, port, "nccl")
    dev = torch.device("cuda", rank)
    cfg, model = load_shard(str(path), rank, 4, dev, torch.float16, layers=4)
    cache = ModelCache(cfg.text, 260, 16, 3, torch.float16, dev, par=Parallel(rank, 4),
                       kv_split=memory.kv_split(cfg, 4, 16))
    g = torch.Generator().manual_seed(2)
    prompt = torch.randint(1000, 60000, (1500,), generator=g).tolist()
    cont = torch.randint(1000, 60000, (4,), generator=g).tolist()
    tables = [list(range(0, 120)), list(range(120, 240))]

    def run(slot, tokens, start):
        step = StepInput.build([SequenceChunk(tokens, start, tables[slot], slot, logits="last")], 16, dev)
        return model.forward(step, cache)[0].float()[-1]

    qwen4_exp.SP_ROWS = 256
    a = [run(0, prompt, 0)]
    qwen4_exp.SP_ROWS = 0
    b = [run(1, prompt, 0)]
    for i in range(4):
        a.append(run(0, [cont[i]], len(prompt) + i))
        b.append(run(1, [cont[i]], len(prompt) + i))
    torch.cuda.synchronize()
    queue.put((rank, [((x - y).abs().max().item(), int(x.argmax()), int(y.argmax())) for x, y in zip(a, b)]))
    distributed.shutdown()


def test_sequence_parallel_prefill_matches_the_replicated_path(flash_next_path):
    from seventy.engine import distributed

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=run_rank, args=(r, port, flash_next_path, queue)) for r in range(4)]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout=900) for _ in range(4))
    for p in procs:
        p.join(timeout=60)
    for diff, top_a, top_b in results[0]:
        assert top_a == top_b and diff < 0.05, results[0]
