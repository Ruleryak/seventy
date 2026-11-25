"""The NVLink all-reduce kernels (csrc/allreduce.cu) on 2, 3 or 4 GPUs: one-shot against a reference, two-shot
against one-shot, and 400 calls in a row with changing row counts mixed with the FP32 all-reduce (slot reuse). Also
the bounded waits (a rank that calls alone gives up and every rank sees it) and the start check (a failure on one
rank turns the kernels off on every rank, and NCCL then runs without peer-to-peer transfers)."""

import os

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.gpu


def run_rank(rank, world, port, queue):
    from seventy.engine import distributed
    from seventy.engine.allreduce import CustomAllReduce

    distributed.share_cpus(world)
    torch.cuda.set_device(rank)
    distributed.init(rank, world, port, "nccl")
    dev = torch.device("cuda", rank)
    ar = CustomAllReduce(rank, world, dev)
    errors = [] if ar.usable else [f"the start check failed: {ar.problem}"]
    for rows in (1, 3, 8, 13, 32, 64):
        g = torch.Generator().manual_seed(rows)
        xs = [(torch.randn(rows, 5120, generator=g) * 2).half() for _ in range(world)]
        res0 = (torch.randn(rows, 5120, generator=g) * 20).half()
        w = torch.rand(5120, generator=g) + 0.5
        total = sum(x.float() for x in xs).half().float()
        want_res = (total + res0.float()).half()
        r = want_res.float()
        want_out = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + 1e-6) * w).half()
        x = xs[rank].to(dev)
        wd = w.to(dev)
        res_a = res0.to(dev)
        out_a = torch.empty_like(x)
        from seventy.ops._lib import check, lib

        stream = torch.cuda.current_stream(dev).cuda_stream
        check("ar_norm", lib().seventy_ar_norm(ar.ctx, out_a.data_ptr(), res_a.data_ptr(), x.data_ptr(), wd.data_ptr(),
                                               rows, 5120, 1e-6, stream))
        res_b = res0.to(dev)
        out_b = torch.empty_like(x)
        check("ar_norm_2shot", lib().seventy_ar_norm_2shot(ar.ctx, out_b.data_ptr(), res_b.data_ptr(), x.data_ptr(),
                                                           wd.data_ptr(), rows, 5120, 1e-6, 1, stream))
        res_c = res0.to(dev)
        out_c = torch.empty_like(x)
        check("ar_norm_2shot", lib().seventy_ar_norm_2shot(ar.ctx, out_c.data_ptr(), res_c.data_ptr(), x.data_ptr(),
                                                           wd.data_ptr(), rows, 5120, 1e-6, 0, stream))
        torch.cuda.synchronize()
        per = -(-rows // world)
        own = slice(rank * per, min(rows, (rank + 1) * per))
        if not torch.equal(res_a.cpu(), want_res):
            errors.append(f"rows {rows}: one-shot residual")
        if (out_a.cpu().float() - want_out.float()).abs().max().item() > 2e-2:
            errors.append(f"rows {rows}: one-shot out")
        if not (torch.equal(out_b, out_a) and torch.equal(res_b, res_a)):
            errors.append(f"rows {rows}: two-shot with gathered residual differs from one-shot")
        if not (torch.equal(out_c, out_a) and torch.equal(res_c[own], res_a[own])):
            errors.append(f"rows {rows}: two-shot (sharded residual) differs from one-shot")
        if not torch.equal(res_c, res_a):
            errors.append(f"rows {rows}: two-shot residual not complete on every rank")
        if not torch.equal(ar.reduce(x).cpu(), total.half()):
            errors.append(f"rows {rows}: FP16 all-reduce differs from the rank-order sum")
    g = torch.Generator().manual_seed(1234)
    for it in range(400):
        rows = (1, 16, 32, 64, 7, 24)[it % 6]
        x = (torch.randn(rows, 5120, generator=g) * 2).half().to(dev) * (rank + 1)
        res0 = (torch.randn(rows, 5120, generator=g) * 20).half().to(dev)
        stream = torch.cuda.current_stream(dev).cuda_stream
        res_a, out_a = res0.clone(), torch.empty_like(x)
        check("ar_norm", lib().seventy_ar_norm(ar.ctx, out_a.data_ptr(), res_a.data_ptr(), x.data_ptr(), wd.data_ptr(),
                                               rows, 5120, 1e-6, stream))
        res_b, out_b = res0.clone(), torch.empty_like(x)
        check("ar_norm_2shot", lib().seventy_ar_norm_2shot(ar.ctx, out_b.data_ptr(), res_b.data_ptr(), x.data_ptr(),
                                                           wd.data_ptr(), rows, 5120, 1e-6, it % 2, stream))
        if it % 5 == 0:
            base = torch.randn(rows, 5120, generator=g).to(dev)
            got = ar.reduce(base * (rank + 1))
            want = base * 1
            for p in range(1, world):
                want = want + base * (p + 1)
            if not torch.equal(got, want):
                errors.append(f"repeat {it}: FP32 all-reduce differs from the rank-order sum")
                break
        if not torch.equal(out_b, out_a):
            errors.append(f"repeat {it}: two-shot out differs from one-shot")
            break
    queue.put((rank, errors))
    distributed.shutdown()


def _peer_gpus() -> bool:
    from seventy.engine.distributed import peer_access

    n = torch.cuda.device_count()
    return n >= 2 and peer_access(4 if n >= 4 else 2)


@pytest.mark.skipif(not _peer_gpus(), reason="needs 2 GPUs with peer-to-peer access (NVLink or PCIe P2P)")
@pytest.mark.parametrize("world", [2, 3, 4])
def test_one_shot_and_two_shot_all_reduce(world):
    from seventy.engine import distributed

    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=run_rank, args=(r, world, port, queue)) for r in range(world)]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout=300) for _ in range(world))
    for p in procs:
        p.join(timeout=60)
    assert all(not e for e in results.values()), results


def run_lone_rank(rank, world, port, queue):
    """Rank 0 calls the all-reduce alone, with a short time limit: the call must end and report it on both ranks."""
    import time

    from seventy.engine import distributed
    from seventy.engine.allreduce import CustomAllReduce
    from seventy.ops._lib import check, lib

    torch.cuda.set_device(rank)
    distributed.init(rank, world, port, "nccl")
    dev = torch.device("cuda", rank)
    ar = CustomAllReduce(rank, world, dev, dim=1024)
    store = distributed.store()
    result = {"usable": ar.usable}
    if rank == 0:
        check("ar_guard", lib().seventy_ar_guard(ar.ctx, None, 300 * 10**6))
        x = torch.ones(4, 1024, dtype=torch.float16, device=dev)
        t0 = time.perf_counter()
        ar.reduce(x)
        torch.cuda.synchronize(dev)
        result["first"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        ar.reduce(x)
        torch.cuda.synchronize(dev)
        result["second"] = time.perf_counter() - t0
        store.set("lone/done", "1")
    else:
        store.wait(["lone/done"])
    result["failure"] = ar.failure()
    queue.put((rank, result))
    distributed.shutdown()


@pytest.mark.skipif(not _peer_gpus(), reason="needs 2 GPUs with peer-to-peer access (NVLink or PCIe P2P)")
def test_wait_gives_up_and_reports():
    from seventy.engine import distributed

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=run_lone_rank, args=(r, 2, port, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout=300) for _ in range(2))
    for p in procs:
        p.join(timeout=60)
    assert results[0]["usable"] and results[1]["usable"]
    assert 0.25 < results[0]["first"] < 5, results
    assert results[0]["second"] < 0.25, results
    for r in (0, 1):
        assert results[r]["failure"] and results[r]["failure"].startswith("rank 0 waited"), results


def run_failed_check(rank, world, port, queue):
    """Rank 1's start check fails: no rank may use the kernels, and NCCL must run without peer-to-peer."""
    from seventy.engine import allreduce, distributed

    if rank == 1:
        allreduce.CustomAllReduce.check = lambda self: "made to fail"
    torch.cuda.set_device(rank)
    distributed.init(rank, world, port, "nccl")
    dev = torch.device("cuda", rank)
    ar = allreduce.CustomAllReduce(rank, world, dev, dim=1024)
    x = torch.full((4, 1024), float(rank + 1), device=dev)
    torch.distributed.all_reduce(x)
    queue.put((rank, (ar.usable, ar.problem, os.environ.get("NCCL_P2P_DISABLE"), x[0, 0].item())))
    distributed.shutdown()


@pytest.mark.skipif(not _peer_gpus(), reason="needs 2 GPUs with peer-to-peer access (NVLink or PCIe P2P)")
def test_failed_check_turns_off_peer_transfers_everywhere(monkeypatch):
    from seventy.engine import distributed

    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = distributed.free_port()
    procs = [ctx.Process(target=run_failed_check, args=(r, 2, port, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout=300) for _ in range(2))
    for p in procs:
        p.join(timeout=60)
    for r in (0, 1):
        usable, problem, p2p_off, total = results[r]
        assert not usable and problem.endswith("GPU 1: made to fail") and p2p_off == "1" and total == 3.0, results

