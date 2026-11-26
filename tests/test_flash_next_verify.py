"""A speculative verify on the Qwen4-Exp GPU path (the first 4 layers of a real checkpoint, flash_next_path: Gated
DeltaNet layers, the PLE layer and a QSA attention layer) leaves the same state as running the accepted tokens one by
one: sequence A decodes six tokens one at a time; sequence B verifies [t1, t2, x, x, x] with three wrong drafts,
commits 2 rows (the Gated DeltaNet and PLE states; the attention and indexer entries of the rejected rows are written
again later), then decodes t3 .. t6. The prompt is longer than the indexer budget, so the sparse selection runs."""

import pytest
import torch

from seventy.checkpoint import Checkpoint
from seventy.config import load_config
from seventy.engine.cache import ModelCache
from seventy.engine.step import SequenceChunk, StepInput
from seventy.ops import kernels_available

pytestmark = [pytest.mark.gpu, pytest.mark.model]
if not (torch.cuda.is_available() and kernels_available()):
    pytestmark = [*pytestmark, pytest.mark.skip(reason="needs a CUDA GPU and the kernel library")]


def test_verify_then_commit_matches_one_by_one(flash_next_path):
    from seventy.model.qwen4_exp import Qwen4ExpTextModel

    cfg = load_config(flash_next_path)
    dev = torch.device("cuda", 0)
    with Checkpoint(flash_next_path) as ckpt:
        model = Qwen4ExpTextModel(cfg, ckpt, dev, torch.float16, layers=4)
    cache = ModelCache(cfg.text, 400, 16, 3, torch.float16, dev)
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(1000, 60000, (2200,), generator=g).tolist()
    cont = torch.randint(1000, 60000, (6,), generator=g).tolist()
    tables = [list(range(0, 180)), list(range(180, 360))]

    def run(slot, tokens, start, logits="last", stash=None):
        step = StepInput.build([SequenceChunk(tokens, start, tables[slot], slot, logits=logits)], 16, dev)
        out, _ = model.forward(step, cache, stash=stash)
        return out.float(), step

    for slot in (0, 1):
        run(slot, prompt, 0)
    n = len(prompt)
    a = [run(0, [cont[i]], n + i)[0][-1] for i in range(6)]
    stash = {}
    verified, step = run(1, [cont[0], cont[1], 7, 8, 9], n, logits="all", stash=stash)
    model.commit(step, cache, stash, torch.tensor([2], dtype=torch.int32, device=dev))
    b = [verified[0], verified[1]] + [run(1, [cont[i]], n + i)[0][-1] for i in range(2, 6)]
    for i in range(6):
        assert int(a[i].argmax()) == int(b[i].argmax()), i
        assert (a[i] - b[i]).abs().max().item() < 0.05, i
