"""Copy weights from host memory to a GPU through pinned buffers.

On some machines, copies to a GPU from pageable host memory are slow (1 to 2 GB/s instead of 6 to 10 GB/s), and
far slower when several processes copy at the same time, as the tensor-parallel processes do when they load the
model: on one rented 4-GPU machine they fell to 0.03 GB/s and the load took 26 minutes. Copies from pinned memory
stayed at 3 GB/s or more there.

to_device copies a tensor in chunks through two pinned buffers: while the GPU reads one buffer, the CPU fills the
other (a plain memcpy in one thread: with PyTorch's threaded copy, the tensor-parallel processes slowed each other
down). A CUDA event per buffer tells when the GPU has read it. Each process keeps the buffers while it loads;
release() frees them. The bytes on the GPU are the same as with Tensor.to.
"""

from __future__ import annotations

import numpy as np
import torch

CHUNK_BYTES = 32 << 20
MIN_BYTES = 1 << 20


class _Stage:
    def __init__(self, device: torch.device):
        self.device = device
        self.buffers = [torch.empty(CHUNK_BYTES, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.arrays = [b.numpy() for b in self.buffers]
        self.events = [None, None]
        self.next = 0

    def wait(self) -> None:
        for e in self.events:
            if e is not None:
                e.synchronize()
        self.events = [None, None]


_stage: _Stage | None = None


def to_device(t: torch.Tensor, device) -> torch.Tensor:
    """A contiguous copy of t on device. Tensors under MIN_BYTES, tensors not in host memory and devices other than
    CUDA use Tensor.to; so does every copy when pinned memory cannot be allocated."""
    global _stage
    device = torch.device(device)
    nbytes = t.numel() * t.element_size()
    if device.type != "cuda" or t.device.type != "cpu" or nbytes < MIN_BYTES:
        return t.to(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if _stage is not None and _stage.device != device:
        release()
    if _stage is None:
        try:
            _stage = _Stage(device)
        except RuntimeError:
            return t.to(device)
    src = t.contiguous().reshape(-1).view(torch.uint8)
    src_np = src.numpy()
    out = torch.empty(tuple(t.shape), dtype=t.dtype, device=device)
    dst = out.reshape(-1).view(torch.uint8)
    stream = torch.cuda.current_stream(device)
    for start in range(0, nbytes, CHUNK_BYTES):
        n = min(CHUNK_BYTES, nbytes - start)
        i = _stage.next
        _stage.next = 1 - i
        if _stage.events[i] is not None:
            _stage.events[i].synchronize()
        buf = _stage.buffers[i]
        np.copyto(_stage.arrays[i][:n], src_np[start:start + n])
        dst[start:start + n].copy_(buf[:n], non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        _stage.events[i] = event
    return out


def release() -> None:
    """Wait for the copies in flight and free the pinned buffers (after the model is loaded)."""
    global _stage
    if _stage is not None:
        _stage.wait()
    _stage = None
