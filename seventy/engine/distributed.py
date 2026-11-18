"""Process group setup for tensor parallel: one process per GPU, NCCL between them (gloo for CPU tests)."""

from __future__ import annotations

import datetime
import socket

import torch
import torch.distributed as dist


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init(rank: int, size: int, port: int, backend: str) -> None:
    dist.init_process_group(backend=backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size,
                            timeout=datetime.timedelta(seconds=600))


def shutdown() -> None:
    """Tear the process group down. Call it on every rank, after dropping CUDA graphs that contain collectives."""
    if dist.is_initialized():
        import gc

        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        dist.destroy_process_group()


def memory_budget(device: torch.device, utilization: float, reserve_bytes: int) -> int:
    """Bytes this GPU can still give to the KV cache and state slots."""
    free, total = torch.cuda.mem_get_info(device)
    return int(total * utilization) - (total - free) - reserve_bytes
