"""The checks of `seventy engine` before it downloads or loads a model: the GPUs (from NVML) and the split of the
model over the GPUs; and the engine's own check of the GPUs."""

from types import SimpleNamespace

import pytest
import torch

from seventy import __main__ as cli
from seventy import memory
from seventy.config import load_config
from seventy.panel import hardware


def inventory(n=4, cc=(7, 0)):
    gpus = [hardware.Gpu(i, "Tesla V100-SXM2-32GB" if cc == (7, 0) else "NVIDIA A100-SXM4-40GB", f"u{i}",
                         f"00000000:{i:02x}:00.0", 32 << 30, cc) for i in range(n)]
    return hardware.Inventory("580", gpus, "cpu", 32, 128 << 30, 100 << 30, 1 << 40, "/")


@pytest.fixture
def gpus(monkeypatch):
    def use(inv, visible):
        monkeypatch.setattr(hardware, "inventory", lambda: inv)
        if visible is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    return use


def test_v100_gpus_pass(gpus):
    gpus(inventory(), "0,1,2,3")
    assert cli._gpu_problems(4) == []
    gpus(inventory(), "2,3")
    assert cli._gpu_problems(2) == []
    gpus(inventory(), None)
    assert cli._gpu_problems(1) == []


def test_other_gpus_are_refused_before_the_download(gpus):
    gpus(inventory(cc=(8, 0)), "0")
    assert any("not a V100" in p for p in cli._gpu_problems(1))


def test_missing_gpu_ids_and_counts(gpus):
    gpus(inventory(), "0,1,2,3,4,5,6,7")
    assert any("no GPU with id 4, 5, 6, 7" in p for p in cli._gpu_problems(8))
    gpus(inventory(), "0,1")
    assert any("only 2 are given" in p for p in cli._gpu_problems(4))
    gpus(inventory(), "0,1,2")
    assert any("1, 2, 4 or 8" in p for p in cli._gpu_problems(3))


def test_unknown_gpu_ids_and_no_nvml_are_left_to_the_engine(gpus):
    gpus(inventory(), "GPU-0a1b")
    assert cli._gpu_problems(1) == []
    broken = inventory()
    broken.error = "NVML is not available"
    gpus(broken, "0")
    assert cli._gpu_problems(1) == []


@pytest.mark.model
def test_split_over_8_gpus_is_refused(model_path):
    cfg = load_config(model_path)
    assert memory.split_problems(cfg, 4) == []
    assert any("4 attention KV heads do not split over 8 GPUs" in p for p in memory.split_problems(cfg, 8))
    assert memory.split_problems(cfg, 3)


def test_engine_stops_on_other_gpus(monkeypatch):
    from seventy.engine import engine

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="NVIDIA A100-SXM4-40GB", major=8, minor=0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(RuntimeError, match=r"GPU 3 \(NVIDIA A100-SXM4-40GB\) has compute capability 8.0"):
        engine._check_gpus([0])
