"""The rendezvous store of rank 0 keeps its port from before the workers start until the process group ends."""

import errno
import socket

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from seventy.engine import distributed


def _worker(rank, size, port, queue):
    distributed.init(rank, size, port, "gloo")
    import torch

    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t)
    queue.put(float(t))
    distributed.shutdown()


def test_master_store_holds_its_port_and_workers_join():
    store = distributed.master_store(2)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(OSError) as e:
            s.bind(("127.0.0.1", store.port))
        assert e.value.errno == errno.EADDRINUSE
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(1, 2, store.port, queue))
    p.start()
    distributed.init(0, 2, store.port, "gloo", store)
    import torch

    t = torch.tensor([1.0])
    dist.all_reduce(t)
    assert float(t) == 3.0 and queue.get(timeout=60) == 3.0
    distributed.shutdown()
    p.join(timeout=60)
    assert p.exitcode == 0
