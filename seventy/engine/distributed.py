"""Process group setup for tensor parallel: one process per GPU, NCCL between them (gloo for CPU tests)."""

from __future__ import annotations

import datetime
import socket

import torch
import torch.distributed as dist


TIMEOUT = datetime.timedelta(seconds=600)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def master_store(size: int) -> dist.TCPStore:
    """The rendezvous store of rank 0, listening from the start on a port that the system chooses (its .port).

    A port from free_port() is free only until the socket closes. Until rank 0 listens on it, another connection
    can take it as its local port (the panel's health checks, or a worker that retries its connection and
    connects to itself), and rank 0 then fails with EADDRINUSE. Rank 0 makes this store before it starts the
    workers, so the port stays taken by the store.
    """
    return dist.TCPStore("127.0.0.1", 0, size, is_master=True, timeout=TIMEOUT, wait_for_workers=False)


def init(rank: int, size: int, port: int, backend: str, store: dist.TCPStore | None = None) -> None:
    """Join the process group: with the store of rank 0 (master_store), else with a TCP store at port."""
    if store is None:
        store = dist.TCPStore("127.0.0.1", port, size, is_master=rank == 0, timeout=TIMEOUT, wait_for_workers=False)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=size, timeout=TIMEOUT)


def shutdown() -> None:
    """Tear the process group down. Call it on every rank, after dropping CUDA graphs that contain collectives."""
    if dist.is_initialized():
        import gc

        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        dist.destroy_process_group()


def memory_budget(device: torch.device, utilization: float, reserve_bytes: int) -> tuple[int, int]:
    """(bytes this GPU can still give to the KV cache and state slots, its total memory). Every rank gets the KV
    cache of the rank with the smallest budget; on GPUs with different memory sizes, that is the smallest GPU."""
    free, total = torch.cuda.mem_get_info(device)
    return int(total * utilization) - (total - free) - reserve_bytes, total
