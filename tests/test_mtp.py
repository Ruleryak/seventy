"""The parts of the MTP drafter that need no model on the GPU: NVFP4 rounding on the GPU, the NVFP4 copy of a packed
LM head, the block table margin and the one-row draft steps past a sequence's blocks, the acceptance counters per draft
position, and the check for an MTP layer in a checkpoint."""

import types

import pytest
import torch

from seventy import nvfp4
from seventy.config import has_mtp
from seventy.engine.engine import Engine
from seventy.engine.runner import ModelRunner
from seventy.model import mtp


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_quantize_on_the_gpu_matches_the_cpu(seed):
    torch.manual_seed(seed)
    w = (torch.randn(1000, 512) * 0.03).half()
    a, b = nvfp4.quantize(w), nvfp4.quantize(w.cuda(), rows=128)
    assert torch.equal(a.scales, b.scales.cpu())
    assert torch.equal(a.packed, b.packed.cpu())
    assert a.global_scale == b.global_scale


@pytest.mark.gpu
def test_nvfp4_head_matches_the_rounded_head():
    from seventy.ops import kernels
    from seventy.ops.kernels import DenseWeight

    torch.manual_seed(1)
    w = (torch.randn(1000, 512) * 0.03).half()
    packed = DenseWeight(w.cuda())
    assert torch.equal(mtp._dense_rows(packed, 0, packed.n_padded)[:1000].cpu(), w)
    x = torch.randn(3, 512).half().cuda()
    ref = x.float() @ nvfp4.dequantize(nvfp4.quantize(w), torch.float16).cuda().float().T
    got = kernels.nvfp4_gemm(x, mtp._nvfp4_head(packed, rows=256)).float()
    assert got.shape == (3, 1000)
    torch.testing.assert_close(got, ref, rtol=0, atol=5e-3)


def test_draft_rows_past_the_blocks_use_the_scratch_block():
    cache = types.SimpleNamespace(block_size=16, scratch_block=99)
    tables = torch.tensor([[3, 4, 99], [7, 99, 99]], dtype=torch.int32)
    positions = torch.tensor([20, 17])
    step = mtp.MtpDrafter.one_row_step(tables, positions, torch.tensor([0, 5]), cache, torch.tensor([1, 2]))
    assert step.kv_slots_t.tolist() == [4 * 16 + 4, 99 * 16 + 1]
    assert step.context_lens_t.tolist() == [20, 17]
    assert step.positions[:, 1].tolist() == [22, 22, 22]
    far = mtp.MtpDrafter.one_row_step(tables, torch.tensor([60]), torch.tensor([0]), cache, torch.tensor([0]))
    assert far.kv_slots_t.tolist() == [99 * 16 + 12]


def test_draft_tables_have_scratch_columns_past_the_longest_table():
    drafter = types.SimpleNamespace(cfg=types.SimpleNamespace(block=5))
    drafter.table_margin = types.MethodType(mtp.MtpDrafter.table_margin, drafter)
    runner = ModelRunner.__new__(ModelRunner)
    runner.drafter, runner.device = drafter, torch.device("cpu")
    runner.cache = types.SimpleNamespace(block_size=16, scratch_block=99)
    t = runner._tables([[1, 2, 3], [4]])
    assert t.shape == (2, 3 + 2)
    assert t[0].tolist() == [1, 2, 3, 99, 99]
    assert t[1].tolist() == [4, 99, 99, 99, 99]


def test_accepted_drafts_are_counted_per_position():
    eng = types.SimpleNamespace(stats={"accepted_tokens": 0, "accepted_at": [0, 0, 0, 0]})
    for n in (0, 2, 4, 1):
        Engine._count_accepted(eng, n)
    assert eng.stats["accepted_tokens"] == 7
    assert eng.stats["accepted_at"] == [3, 2, 1, 1]


def test_the_test_model_has_an_mtp_layer(model_path):
    assert has_mtp(model_path)


def test_an_mtp_layer_is_found_in_a_single_file_checkpoint(tmp_path):
    from safetensors.torch import save_file

    names = ("mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight")
    save_file({n: torch.zeros(2) for n in names}, str(tmp_path / "model.safetensors"))
    assert has_mtp(tmp_path)
    save_file({names[0]: torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    assert not has_mtp(tmp_path)
    assert not has_mtp(tmp_path / "missing")
