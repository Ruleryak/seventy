"""MTP drafter: the multi-token prediction layer of the target checkpoint (mtp.*, Qwen3.5 / 3.6 / 3.8). It drafts k
tokens one after the other (greedy) and shares the target's embeddings and LM head.

One pass over rows at positions p: e = rms_norm(embed(token at p + 1), pre_fc_norm_embedding), h = rms_norm(hidden
at p, pre_fc_norm_hidden), x = fc([e, h]), then one decoder layer (full attention with an output gate and its own
KV cache at layer index num_layers, then the MLP), then rms_norm(x, mtp.norm). That output predicts the token at
p + 2 (through the LM head), and the next pass takes it as its hidden. The hidden at p of the first pass is the
target's final-normed output at p (the LM head's input).

After a speculative verify of [anchor, d1 .. dk] at positions P .. P + k with n accepted drafts, a catch-up pass
over all k + 1 rows (row i: the token at P + i + 1, hidden: the target's output of row i) writes the MTP keys and
values at P .. P + k. Rows past n are written again by the next step before anything reads them. The output of row
n gives d1 (the token at P + n + 2), and k - 1 one-row passes at P + n + 1 .. P + n + k - 1 give d2 .. dk. After a
prefill, one pass over the prompt rows (token: the next prompt token) writes the MTP keys and values of the
prompt; the first draft is one pass over the last prompt row with the sampled token, then k - 1 one-row passes.

The MTP layer's dense weights are rounded to NVFP4 for the NVFP4 GEMM (a quarter of the FP16 weight traffic of each
draft pass; SEVENTY_MTP_NVFP4=0 keeps them FP16), and the drafts take the argmax of an NVFP4 copy of the target's LM
head shard (SEVENTY_MTP_HEAD=fp16 uses the head itself). Greedy on the release prompts, the NVFP4 head accepts as many
drafts as an FP8 copy and as an NVFP4 head whose top 32 candidates are scored again in FP16, and it is the fastest.

The drafts are greedy: the draft distribution is one-hot, so lossless rejection sampling accepts a draft with the
target's probability of it, and the output distribution stays the target's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import ops
from ..checkpoint import Checkpoint
from ..engine.step import StepInput
from ..parallel import SINGLE, Parallel
from .layers import Linear, Source
from .qwen3_5 import FINAL, MLP, AttentionLayer, _norm_weight


@dataclass(frozen=True)
class MtpConfig:
    block: int
    layer: int
    top_k: int = 1
    window: int = 1 << 30
    target_layers: tuple[int, ...] = (FINAL,)


@dataclass
class Draft:
    tokens: torch.Tensor
    probs: torch.Tensor
    candidates: torch.Tensor


def _dense_rows(w, start: int, stop: int) -> torch.Tensor:
    """Rows [start, stop) (multiples of 32) of a packed FP16 DenseWeight as a dense FP16 tensor."""
    from ..ops.kernels import _LANES

    inv = torch.empty(32, dtype=torch.long)
    inv[torch.tensor(_LANES)] = torch.arange(32)
    blk = w.wpk[start // 32 : stop // 32].permute(0, 2, 1, 3)[:, inv.to(w.wpk.device)]
    return blk.reshape(-1, w.k)


def _nvfp4_head(w, rows: int = 1024):
    """An NVFP4 copy of a packed FP16 LM head shard (one global scale), packed for the NVFP4 GEMM."""
    from .. import nvfp4

    n = w.n
    amax = max(_dense_rows(w, s, min(s + rows, w.n_padded)).float().abs().amax() for s in range(0, w.n_padded, rows))
    scale = float(448.0 * 6.0 / amax)
    parts = [nvfp4.quantize(_dense_rows(w, s, min(s + rows, w.n_padded))[: max(0, min(rows, n - s))], scale)
             for s in range(0, w.n_padded, rows) if s < n]
    return nvfp4.pack_for_gemm(parts)


class MtpDrafter:
    def __init__(self, path: str | Path, target, device, dtype=torch.float16, par: Parallel = SINGLE,
                 tokens: int = 4):
        t = target.cfg
        self.target, self.par, self.device, self.dtype = target, par, torch.device(device), dtype
        self.cfg = MtpConfig(block=tokens + 1, layer=t.num_layers)
        self.eps = t.rms_norm_eps
        with Checkpoint(Path(path)) as ckpt:
            if "mtp.fc.weight" not in ckpt:
                raise ValueError(f"{path} has no MTP layer (mtp.fc.weight)")
            q = os.environ.get("SEVENTY_MTP_NVFP4", "1") != "0"
            self.fc = Linear(ckpt, [Source("mtp.fc")], dtype, device, quantize=q)
            self.attn = AttentionLayer(t, ckpt, "mtp.layers.0.self_attn", par, dtype, device, False, quantize=q)
            self.mlp = MLP(t, ckpt, "mtp.layers.0.mlp", par, dtype, device, False, quantize=q)
            self.input_norm = _norm_weight(ckpt, "mtp.layers.0.input_layernorm.weight", device, 1.0)
            self.post_norm = _norm_weight(ckpt, "mtp.layers.0.post_attention_layernorm.weight", device, 1.0)
            self.pre_e = _norm_weight(ckpt, "mtp.pre_fc_norm_embedding.weight", device, 1.0)
            self.pre_h = _norm_weight(ckpt, "mtp.pre_fc_norm_hidden.weight", device, 1.0)
            self.norm = _norm_weight(ckpt, "mtp.norm.weight", device, 1.0)
        self.attn.inv_freq = target.inv_freq
        self.head = None
        if (self.device.type == "cuda" and dtype == torch.float16 and target.lm_head_packed is not None
                and os.environ.get("SEVENTY_MTP_HEAD", "nvfp4") != "fp16"):
            self.head = _nvfp4_head(target.lm_head_packed)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def allocate(self, num_slots: int) -> None:
        """The MTP keys and values live in the KV cache blocks (layer cfg.layer); no state per slot."""

    def copy_slot(self, src: int, dst: int) -> None:
        """No state per slot (see allocate)."""

    def ring_bytes_per_slot(self) -> int:
        return 0

    def table_margin(self, block_size: int) -> int:
        """Block table columns past a sequence's blocks that the draft passes may reach (positions up to two
        verify blocks past the anchor): they hold the scratch block."""
        return -(-2 * self.cfg.block // block_size) + 1

    def run(self, tokens: torch.Tensor, hidden: torch.Tensor, step: StepInput, cache) -> torch.Tensor:
        """One MTP pass over the rows of step: tokens are the tokens one position after each row, hidden the hidden
        state at each row. Writes the MTP keys and values of the rows; returns the normed output (all rows)."""
        e = ops.rms_norm(self.target.embed_tokens(tokens), self.pre_e, self.eps)[0]
        h = ops.rms_norm(hidden.to(self.dtype), self.pre_h, self.eps)[0]
        residual = self.fc(torch.cat((e, h), dim=-1))
        x = ops.rms_norm(residual, self.input_norm, self.eps)[0]
        if x.is_cuda:
            rope = slots = None
        else:
            rope = ops.rotary_cos_sin(step.positions, self.target.inv_freq, self.target.cfg.mrope_section)
            slots = ops.kv_slots(step.query_start, step.context_lens, step.block_tables, step.block_size)
        x = self.attn(x, step, cache, self.cfg.layer, rope, slots)
        x, residual = self.par.reduce_norm(x, self.post_norm, self.eps, residual)
        x = self.mlp(x)
        out, _ = self.par.reduce_norm(x, self.norm, self.eps, residual)
        return out

    def argmax(self, h: torch.Tensor) -> torch.Tensor:
        """The greedy token of each row of MTP outputs h, from the draft head (the NVFP4 copy of the LM head)."""
        return self.target.head_top(h, 1, self.head)[0].long()

    @staticmethod
    def one_row_step(tables: torch.Tensor, positions: torch.Tensor, delta: torch.Tensor, cache,
                     slots: torch.Tensor) -> StepInput:
        """A step of one row per sequence at positions (device tensors only, for CUDA graphs). A position past the
        sequence's blocks (a draft past the verify block) writes to the scratch block: the next catch-up pass
        writes it again in the sequence's own block before anything reads it."""
        b, dev, bs = positions.shape[0], positions.device, cache.block_size
        col = (positions // bs).long()
        width = tables.shape[1]
        blk = tables.gather(1, col.clamp(max=width - 1)[:, None]).long()[:, 0]
        blk = torch.where(col < width, blk, torch.full_like(blk, cache.scratch_block))
        kv = blk * bs + positions % bs
        return StepInput(
            token_ids=positions, positions=(positions + delta).reshape(1, -1).expand(3, -1).contiguous(),
            block_tables=tables, logit_rows=torch.arange(b, device=dev), query_start=list(range(b + 1)),
            context_lens=[0] * b, state_slots=[0] * b, block_size=bs,
            query_start_t=torch.arange(b + 1, dtype=torch.int32, device=dev), context_lens_t=positions.int(),
            state_slots_t=slots.int(), kv_slots_t=kv.int(), max_blocks=0,
        )

    def chain(self, h: torch.Tensor, first: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor,
              delta: torch.Tensor, slots: torch.Tensor, cache) -> torch.Tensor:
        """Drafts [b, k]: first (the draft from h) and k - 1 more one-row passes, the first at positions."""
        drafts = [first]
        k = self.cfg.block - 1
        for j in range(1, k):
            step = self.one_row_step(tables, positions + (j - 1), delta, cache, slots)
            h = self.run(drafts[-1], h, step, cache)
            drafts.append(self.argmax(h))
        return torch.stack(drafts, dim=1)

    def after_rows(self, step: StepInput, hidden: torch.Tensor, row_tokens: torch.Tensor, last_rows: torch.Tensor,
                   positions: torch.Tensor, tables: torch.Tensor, delta: torch.Tensor, slots: torch.Tensor,
                   cache) -> Draft:
        """The next drafts after a verify step: one MTP pass over all its rows (row_tokens: the token after each
        row; hidden: the target's final-normed rows), then the chain from each sequence's last accepted row
        (last_rows), with the next pass at positions."""
        out = self.run(row_tokens, hidden, step, cache)
        h = out[last_rows]
        return self.one_hot(self.chain(h, self.argmax(h), positions, tables, delta, slots, cache))

    def after_verify(self, step: StepInput, hidden: torch.Tensor, new: torch.Tensor, n: torch.Tensor,
                     starts: torch.Tensor, tables: torch.Tensor, delta: torch.Tensor, slots: torch.Tensor,
                     cache) -> Draft:
        """after_rows for a verify step of block rows per sequence (CUDA graphs): new [b, block] (the accepted
        drafts and the new token: the token after each row), n [b] (accepted drafts), starts [b] (anchor
        positions)."""
        b, blk = new.shape
        rows = torch.arange(b, device=new.device) * blk + n
        return self.after_rows(step, hidden, new.reshape(-1), rows, starts + n + 1, tables, delta, slots, cache)

    def first(self, hidden: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor, tables: torch.Tensor,
              delta: torch.Tensor, slots: torch.Tensor, cache) -> Draft:
        """The first drafts after a prefill: hidden [b, H] (the target's output at the last prompt row),
        tokens [b] (the sampled tokens), positions [b] (of the last prompt rows)."""
        step = self.one_row_step(tables, positions, delta, cache, slots)
        h = self.run(tokens, hidden, step, cache)
        d = self.chain(h, self.argmax(h), positions + 1, tables, delta, slots, cache)
        return self.one_hot(d)

    @staticmethod
    def one_hot(d: torch.Tensor) -> Draft:
        return Draft(d, torch.ones(*d.shape, 1, dtype=torch.float32, device=d.device), d[..., None])
