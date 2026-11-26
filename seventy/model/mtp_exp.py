"""MTP drafter of Qwen4-Exp models (Qwen3.8 Flash-Next): the checkpoint's MTP layer (mtp.*), drafting k tokens one
after the other (greedy), with the target's embeddings and LM head.

The layer works on hyper-connection streams. One pass over rows at positions p: e = fc_embedding(rms_norm(embed(token
at p + 1), pre_fc_norm_embedding)); the streams [rows, hc x hidden] at p get an RMS norm per stream
(pre_fc_norm_hidden, one weight of hc x hidden; SEVENTY_MTP_HIDDEN_NORM=full takes one norm over all streams) and
fc_hidden per stream, and e adds to every stream. Then one Qwen4-Exp layer (hyper-connections around a QSA attention
layer with its own KV cache, indexer and pooled keys at layer index num_layers, and a mixture of experts; no PLE).
The streams after the layer are the next pass's input; their final mix (hyper_connection_mixer, without injection)
feeds the LM head and predicts the token at p + 2. The streams of the first pass are the target's streams after its
last layer, before its final mix (capture STREAMS).

The engine uses the passes as for MtpDrafter (seventy/model/mtp.py): after a verify, a catch-up pass over all its
rows, then one-row passes; after a prefill, one pass over the prompt rows. The drafts take the argmax of an NVFP4
copy of the target's LM head shard (SEVENTY_MTP_HEAD=fp16 uses the head itself)."""

from __future__ import annotations

import os

import torch

from ..checkpoint import Checkpoint
from ..engine.step import StepInput
from ..parallel import SINGLE, Parallel
from .layers import Linear, Source
from .mtp import MtpConfig, MtpDrafter, _nvfp4_head
from .qwen4_exp import STREAMS, ExpLayer, HyperConnection, _grouped_norm_weight, grouped_rms_norm


class ExpMtpDrafter(MtpDrafter):
    def __init__(self, path, target, device, dtype=torch.float16, par: Parallel = SINGLE,
                 tokens: int = 4):
        t = target.cfg
        self.target, self.par, self.device, self.dtype = target, par, torch.device(device), dtype
        self.cfg = MtpConfig(block=tokens + 1, layer=t.num_layers, target_layers=(STREAMS,))
        self.eps = t.rms_norm_eps
        self.hc, self.hidden = t.exp.hc_count, t.hidden_size
        self.full_norm = os.environ.get("SEVENTY_MTP_HIDDEN_NORM", "grouped") == "full"
        eos = t.eos_token_ids[0] if t.eos_token_ids else 0
        with Checkpoint(path) as ckpt:
            if "mtp.fc_hidden.weight" not in ckpt:
                raise ValueError(f"{path} has no Qwen4-Exp MTP layer (mtp.fc_hidden.weight)")
            self.fc_e = Linear(ckpt, [Source("mtp.fc_embedding")], dtype, device)
            self.fc_h = Linear(ckpt, [Source("mtp.fc_hidden")], dtype, device)
            self.pre_e = _grouped_norm_weight(ckpt, "mtp.pre_fc_norm_embedding.weight", device)
            self.pre_h = _grouped_norm_weight(ckpt, "mtp.pre_fc_norm_hidden.weight", device)
            self.layer = ExpLayer(t, ckpt, t.num_layers, par, dtype, device, False, eos, prefix="mtp.layers.0",
                                  kind="full_attention")
            self.mixer = HyperConnection(t, ckpt, "mtp.hyper_connection_mixer", dtype, device, combine=False)
        attn = self.layer.mixer.attn
        attn.inv_freq, attn.rope_scale = target.inv_freq, target.rope_scale
        self.attn = attn
        self.head = None
        if (self.device.type == "cuda" and dtype == torch.float16 and target.lm_head_packed is not None
                and os.environ.get("SEVENTY_MTP_HEAD", "nvfp4") != "fp16"):
            from ..nvfp4 import GemmWeight

            packed = target.lm_head_packed
            self.head = packed if isinstance(packed, GemmWeight) else _nvfp4_head(packed)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run(self, tokens: torch.Tensor, hidden: torch.Tensor, step: StepInput, cache) -> torch.Tensor:
        """One MTP pass over the rows of step: tokens are the tokens one position after each row, hidden the streams
        FP32 [rows, hc x hidden] at each row. Writes the MTP layer's keys, values and indexer keys of the rows;
        returns the streams after the layer (all rows)."""
        t, target = tokens.shape[0], self.target
        par = target.par
        e = grouped_rms_norm(target.embed_tokens(tokens).float(), self.pre_e, self.hidden, self.eps)
        e = self.fc_e(e.to(self.dtype)).float()
        group = self.hc * self.hidden if self.full_norm else self.hidden
        hn = grouped_rms_norm(hidden.float(), self.pre_h, group, self.eps).to(self.dtype)
        hf = self.fc_h(hn.view(t * self.hc, self.hidden)).float().view(t, self.hc, self.hidden)
        h = (hf + e[:, None, :]).reshape(t, -1)
        if h.is_cuda:
            pos = step.positions.to(torch.float32)[target.rope_axis].T
            angles = pos * target.inv_freq
            rope = (torch.cos(angles) * target.rope_scale, torch.sin(angles) * target.rope_scale)
            slots = step.kv_slots_t.long()
        else:
            from .. import ops

            cos, sin = ops.rotary_cos_sin(step.positions, target.inv_freq, target.cfg.mrope_section)
            rope = (cos * target.rope_scale, sin * target.rope_scale)
            slots = ops.kv_slots(step.query_start, step.context_lens, step.block_tables, step.block_size)
        cache.positions.view(-1, 3)[slots] = step.positions.T.to(cache.positions.dtype)
        layer = self.layer
        x, inject = layer.attn_hc(h)
        y = layer.mixer(x.to(self.dtype), step, cache, self.cfg.layer, rope, slots, target)
        h = target._inject(h, par.all_reduce(y), inject)
        x, inject = layer.mlp_hc(h)
        return target._inject(h, par.all_reduce(layer.mlp(x.to(self.dtype))), inject)

    def argmax(self, h: torch.Tensor) -> torch.Tensor:
        """The greedy token of each row of MTP output streams h: the final mix, then the draft head."""
        x, _ = self.mixer(h)
        return self.target.head_top(x.to(self.dtype), 1, self.head)[0].long()
