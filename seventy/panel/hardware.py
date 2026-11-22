"""The machine: GPUs and their NVLink links (NVML), CPU, RAM and disk. No PyTorch.

GPU ids are in PCI bus order, the order CUDA uses with CUDA_DEVICE_ORDER=PCI_BUS_ID (the engine sets it), so an
id in the panel is the id the engine gets in CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

_lock = threading.Lock()
_ready = False


def _nvml():
    global _ready
    import pynvml

    with _lock:
        if not _ready:
            pynvml.nvmlInit()
            _ready = True
    return pynvml


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass
class Gpu:
    id: int
    name: str
    uuid: str
    bus_id: str
    memory_total: int
    compute_capability: tuple[int, int]
    links: dict[int, int] = field(default_factory=dict)


@dataclass
class Inventory:
    driver: str
    gpus: list[Gpu]
    cpu: str
    cpu_threads: int
    ram_total: int
    ram_available: int
    disk_free: int
    disk_path: str
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for g in d["gpus"]:
            g["links"] = {str(k): v for k, v in g["links"].items()}
        return d


def _meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def gpus() -> list[Gpu]:
    """The GPUs in PCI bus order, with NVLink link counts to each other GPU."""
    nv = _nvml()
    found = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        found.append((_text(nv.nvmlDeviceGetPciInfo(h).busId), h))
    found.sort(key=lambda x: x[0].lower())
    bus_to_id = {bus.lower(): i for i, (bus, _) in enumerate(found)}
    out = []
    for i, (bus, h) in enumerate(found):
        links: dict[int, int] = {}
        for link in range(18):
            try:
                if nv.nvmlDeviceGetNvLinkState(h, link) != 1:
                    continue
                remote = _text(nv.nvmlDeviceGetNvLinkRemotePciInfo(h, link).busId).lower()
            except nv.NVMLError:
                continue
            j = bus_to_id.get(remote)
            if j is not None:
                links[j] = links.get(j, 0) + 1
        out.append(Gpu(i, _text(nv.nvmlDeviceGetName(h)), _text(nv.nvmlDeviceGetUUID(h)), bus,
                       int(nv.nvmlDeviceGetMemoryInfo(h).total), tuple(nv.nvmlDeviceGetCudaComputeCapability(h)),
                       links))
    return out


def driver_version() -> str:
    """The version of the NVIDIA driver, for example 580.178.04 (raises an exception when NVML is missing)."""
    return _text(_nvml().nvmlSystemGetDriverVersion())


def inventory(disk_path: str | None = None) -> Inventory:
    mem = _meminfo()
    path = disk_path or os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface")
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe).free if probe.exists() else 0
    try:
        driver, found, error = driver_version(), gpus(), None
    except Exception as e:
        driver, found, error = "", [], f"NVML is not available: {e}"
    return Inventory(driver, found, _cpu_name(), os.cpu_count() or 0, mem.get("MemTotal", 0),
                     mem.get("MemAvailable", 0), disk, str(probe), error)


def gpu_set_problems(inv: Inventory, ids: list[int]) -> list[str]:
    """Why a set of GPU ids cannot run one tensor-parallel engine (empty when it can)."""
    problems = []
    known = {g.id: g for g in inv.gpus}
    if not ids:
        return ["choose at least one GPU"]
    if len(set(ids)) != len(ids):
        problems.append("a GPU is listed twice")
    missing = [i for i in ids if i not in known]
    if missing:
        problems.append(f"no GPU with id {', '.join(map(str, missing))}")
        return problems
    if len(ids) not in (1, 2, 4, 8):
        problems.append("tensor parallel needs 1, 2, 4 or 8 GPUs")
    for i in ids:
        if known[i].compute_capability != (7, 0):
            problems.append(f"GPU {i} ({known[i].name}) is not a V100 (sm70)")
    return problems


def nvlink_full(inv: Inventory, ids: list[int]) -> bool:
    """Whether every pair of the GPUs has a direct NVLink link."""
    known = {g.id: g for g in inv.gpus}
    return all(known[a].links.get(b, 0) > 0 for a in ids for b in ids if a != b and a in known and b in known)


@dataclass
class GpuSample:
    id: int
    util: int
    memory_used: int
    temperature: int
    power: float
    power_limit: float
    sm_clock: int


def sample() -> list[GpuSample]:
    """Live GPU readings in PCI bus order."""
    nv = _nvml()
    handles = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        handles.append((_text(nv.nvmlDeviceGetPciInfo(h).busId).lower(), h))
    handles.sort(key=lambda x: x[0])
    out = []
    for i, (_, h) in enumerate(handles):
        def read(fn, *args, default=0):
            try:
                return fn(h, *args)
            except nv.NVMLError:
                return default

        util = read(nv.nvmlDeviceGetUtilizationRates, default=None)
        mem = read(nv.nvmlDeviceGetMemoryInfo, default=None)
        out.append(GpuSample(i, int(util.gpu) if util else 0, int(mem.used) if mem else 0,
                             int(read(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU)),
                             read(nv.nvmlDeviceGetPowerUsage) / 1000.0, read(nv.nvmlDeviceGetEnforcedPowerLimit) / 1000.0,
                             int(read(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM))))
    return out
