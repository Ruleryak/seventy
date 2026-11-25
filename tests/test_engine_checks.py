"""The checks of `seventy engine` before it downloads or loads a model: the GPUs (from NVML), the split of the
model over the GPUs, and the memory of the weights (with the head copy of tied embeddings); the engine's own check of
the GPUs; the message when the KV cache does not fit; and the CPU threads of each engine process."""

import os
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
    assert cli._gpu_problems(3) == []
    gpus(inventory(8), "0,1,2,3,4,5")
    assert any("1, 2, 3, 4 or 8" in p for p in cli._gpu_problems(6))


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
    assert memory.split_problems(cfg, 3) == []


def test_engine_stops_on_other_gpus(monkeypatch):
    from seventy.engine import engine

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="NVIDIA A100-SXM4-40GB", major=8, minor=0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(RuntimeError, match=r"GPU 3 \(NVIDIA A100-SXM4-40GB\) has compute capability 8.0"):
        engine._check_gpus([0])


def test_usable_cpus_follows_a_cgroup_quota(tmp_path):
    from seventy.engine import distributed

    cores = len(os.sched_getaffinity(0))
    assert distributed.usable_cpus(str(tmp_path)) == cores
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert distributed.usable_cpus(str(tmp_path)) == cores
    (tmp_path / "cpu.max").write_text("150000 100000\n")
    assert distributed.usable_cpus(str(tmp_path)) == min(cores, 1)
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("200000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert distributed.usable_cpus(str(tmp_path)) == min(cores, 2)
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    assert distributed.usable_cpus(str(tmp_path)) == cores


def test_share_cpus_splits_the_cpus_and_keeps_omp_num_threads(monkeypatch):
    from seventy.engine import distributed

    before = torch.get_num_threads()
    try:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        distributed.share_cpus(4)
        assert torch.get_num_threads() == max(1, distributed.usable_cpus() // 4)
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        torch.set_num_threads(5)
        distributed.share_cpus(4)
        assert torch.get_num_threads() == 5
    finally:
        torch.set_num_threads(before)


def tiny_config(tied: bool, vocab: int = 1 << 20):
    """A small Qwen3.5 text config (hidden 1024) with a large vocabulary, with or without tied embeddings."""
    from seventy.config import parse_config

    t = {"model_type": "qwen3_5_text", "attn_output_gate": True, "head_dim": 256, "hidden_act": "silu",
         "hidden_size": 1024, "intermediate_size": 4096, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
         "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
         "linear_num_value_heads": 8, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
         "num_attention_heads": 8, "num_hidden_layers": 4, "num_key_value_heads": 4, "rms_norm_eps": 1e-6,
         "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                             "mrope_section": [11, 11, 10], "mrope_interleaved": True},
         "vocab_size": vocab, "dtype": "bfloat16", "tie_word_embeddings": tied}
    return parse_config({"architectures": ["Qwen3_5ForCausalLM"], "text_config": t})


def test_tied_embeddings_add_the_head_copy():
    assert memory.tied_head_bytes(tiny_config(True), 1) == (1 << 20) * 1024 * 2
    assert memory.tied_head_bytes(tiny_config(True), 4) == (1 << 20) * 1024 * 2 // 4
    assert memory.tied_head_bytes(tiny_config(False), 1) == 0


def test_weights_check_counts_the_head_copy_of_tied_embeddings(monkeypatch):
    from seventy.engine import engine

    monkeypatch.setattr(memory, "checkpoint_bytes", lambda path: 15 << 30)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=16 << 30))
    config = SimpleNamespace(model="m", draft_model=None)
    engine._check_weights_fit(config, [0], tiny_config(False))
    with pytest.raises(RuntimeError, match=r"the model needs about 17.0 GiB per GPU on 1 GPU\(s\)"):
        engine._check_weights_fit(config, [0], tiny_config(True))


def test_kv_shortage_names_the_numbers_and_the_options():
    short = memory.kv_shortage(2 << 30, 3 << 30, 8, 24)
    assert "2.0 GiB is left after the weights and the activation reserve" in short
    assert "the state of 8 streams and 24 prefix-cache entries takes 3.0 GiB" in short
    assert "--max-num-seqs" in short and "--no-prefix-cache" in short and "--gpus" in short
    over = memory.kv_shortage(-(1 << 29), 1 << 30, 8, 24)
    assert "take 0.5 GiB more than the budget of the smallest GPU" in over and "--max-num-seqs" in over
