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
