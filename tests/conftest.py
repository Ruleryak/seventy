import os
from pathlib import Path

import pytest
import torch

from seventy.ops import kernels_available

MODEL_REPO = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"


def pytest_collection_modifyitems(config, items):
    gpu_ok = torch.cuda.is_available() and kernels_available()
    for item in items:
        if "gpu" in item.keywords and not gpu_ok:
            item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU and the kernel library"))


def available_memory() -> int:
    """Bytes of host memory that the tests can still use: MemAvailable, or less under a cgroup memory limit (as in a
    container)."""
    fields = dict(row.split(":", 1) for row in Path("/proc/meminfo").read_text().splitlines() if ":" in row)
    avail = int(fields.get("MemAvailable", "0 kB").split()[0]) * 1024
    for limit_file, usage_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                                    "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit = Path(limit_file).read_text().strip()
            usage = int(Path(usage_file).read_text())
        except (OSError, ValueError):
            continue
        if limit != "max" and int(limit) < 1 << 60:
            avail = min(avail, int(limit) - usage)
        break
    return avail


def _find_model() -> Path | None:
    override = os.environ.get("SEVENTY_TEST_MODEL")
    if override:
        return Path(override)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(MODEL_REPO, local_files_only=True))
    except Exception:
        return None


@pytest.fixture(scope="session")
def model_path() -> Path:
    path = _find_model()
    if path is None or not (path / "config.json").is_file():
        pytest.skip(f"model checkpoint not found (set SEVENTY_TEST_MODEL or download {MODEL_REPO})")
    return path
