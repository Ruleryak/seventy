"""Process group setup for tensor parallel: one process per GPU, NCCL between them (gloo for CPU tests)."""

from __future__ import annotations

import datetime
import os
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


_store: dist.Store | None = None


def init(rank: int, size: int, port: int, backend: str, store: dist.TCPStore | None = None) -> None:
    """Join the process group: with the store of rank 0 (master_store), else with a TCP store at port. NCCL makes
    its communicator at the first collective, so environment settings for NCCL still take effect until then."""
    global _store
    if store is None:
        store = dist.TCPStore("127.0.0.1", port, size, is_master=rank == 0, timeout=TIMEOUT, wait_for_workers=False)
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=size, timeout=TIMEOUT)
    _store = store


def store() -> dist.Store:
    """The key-value store of the process group (TCP, through rank 0): for exchanges that must not use NCCL."""
    if _store is not None:
        return _store
    return dist.distributed_c10d._get_default_store()


def shutdown() -> None:
    """Tear the process group down. Call it on every rank, after dropping CUDA graphs that contain collectives."""
    global _store
    _store = None
    if dist.is_initialized():
        import gc

        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        dist.destroy_process_group()


def usable_cpus(cgroup: str = "/sys/fs/cgroup") -> int:
    """CPUs this process may run on: its affinity mask, or fewer under a CPU quota of its cgroup (v2 cpu.max, or v1
    cpu.cfs_quota_us), as in a container."""
    n = len(os.sched_getaffinity(0))
    try:
        with open(os.path.join(cgroup, "cpu.max")) as f:
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, int(quota) // int(period))
    except (OSError, ValueError):
        try:
            with open(os.path.join(cgroup, "cpu", "cpu.cfs_quota_us")) as f:
                quota = int(f.read())
            with open(os.path.join(cgroup, "cpu", "cpu.cfs_period_us")) as f:
                period = int(f.read())
            if quota > 0:
                n = min(n, quota // period)
        except (OSError, ValueError):
            pass
    return max(1, n)


def share_cpus(size: int) -> None:
    """Give each of the size engine processes an equal share of the usable CPUs for PyTorch's CPU threads (unless
    OMP_NUM_THREADS is set). PyTorch's default counts the cores of the machine and ignores a CPU quota; the
    tensor-parallel processes then run many more threads than CPUs, and on a busy host each multithreaded CPU
    operation (weight conversions at load time) waits for its slowest thread. On a rented 4-GPU machine with 80
    cores, a quota of 38 CPUs and other load, small CPU operations took minutes."""
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(max(1, usable_cpus() // size))


def _cpu_list(text: str) -> set[int]:
    cpus = set()
    for part in text.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def gpu_numa_node(index: int) -> int | None:
    """The NUMA node of CUDA device index (from its PCI address in sysfs), or None when unknown."""
    p = torch.cuda.get_device_properties(index)
    path = f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0/numa_node"
    try:
        with open(path) as f:
            node = int(f.read())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def bind_numa(device: torch.device, size: int) -> int | None:
    """Run this process on the CPUs of its GPU's NUMA node (on hosts with more than one node), so that its pinned
    host memory (offloaded experts, the PLE tables) is allocated on the node next to the GPU: with 4 V100s copying
    from host memory at once, each copy runs at 9.3 GB/s from local memory and at 4.7 to 6.4 GB/s when the memory of
    all GPUs is on one node. The process's PyTorch CPU threads become an equal share of the node's CPUs among the
    size GPUs on that node. Call it after starting the other ranks' processes, which would inherit the binding.
    SEVENTY_NUMA=0 turns it off. Returns the node."""
    if os.environ.get("SEVENTY_NUMA", "1") == "0" or device.type != "cuda":
        return None
    try:
        nodes = [n for n in os.listdir("/sys/devices/system/node") if n.startswith("node") and n[4:].isdigit()]
    except OSError:
        return None
    node = gpu_numa_node(device.index)
    if len(nodes) < 2 or node is None:
        return None
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            cpus = _cpu_list(f.read()) & os.sched_getaffinity(0)
    except OSError:
        return None
    if not cpus:
        return None
    os.sched_setaffinity(0, cpus)
    if "OMP_NUM_THREADS" not in os.environ:
        peers = sum(1 for i in range(size) if gpu_numa_node(i) == node) or 1
        torch.set_num_threads(max(1, usable_cpus() // peers))
    return node


def peer_access(size: int) -> bool:
    """Whether each pair of the first size CUDA devices can map the other's memory (NVLink, or PCIe peer-to-peer).
    The custom all-reduce needs it; without it every all-reduce goes through NCCL. All ranks get the same answer."""
    return all(torch.cuda.can_device_access_peer(a, b) for a in range(size) for b in range(size) if a != b)


def memory_budget(device: torch.device, utilization: float, reserve_bytes: int) -> tuple[int, int]:
    """(bytes this GPU can still give to the KV cache and state slots, its total memory). Every rank gets the KV
    cache of the rank with the smallest budget; on GPUs with different memory sizes, that is the smallest GPU."""
    free, total = torch.cuda.mem_get_info(device)
    return int(total * utilization) - (total - free) - reserve_bytes, total
