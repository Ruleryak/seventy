"""The Qwen3.5-family text model: Gated DeltaNet layers and gated full-attention layers in one residual stream.

Weights are loaded as this rank's tensor-parallel shard:
- Attention: query heads, KV heads and the o_proj input columns split over the ranks.
- Gated DeltaNet: key heads, value heads (with their convolution channels, z, a and b rows) and the out_proj
  input columns split over the ranks.
- MLP: the intermediate size split over the ranks.
- Embedding and LM head: the vocabulary split over the ranks.
"""

from __future__ import annotations

import math

import torch

from .. import ops
from ..checkpoint import Checkpoint
from ..config import ModelConfig, TextConfig
from ..engine.cache import ModelCache
from ..engine.spec import merge_top
from ..engine.step import StepInput
from ..ops import gdn_prefill, kernels
from ..parallel import SINGLE, Parallel

RECURRENT_MAX_TOKENS = 32
from .layers import Linear, Source, _to_dtype


def _norm_weight(ckpt: Checkpoint, name: str, device, offset: float) -> torch.Tensor:
    return (ckpt.tensor(name).float() + offset).to(device)


class AttentionLayer:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, par: Parallel, dtype, device, keep_dense):
        d = cfg.head_dim
        h0, h1 = par.span(cfg.num_heads)
        k0, k1 = par.span(cfg.num_kv_heads)
        self.heads, self.kv_heads, self.head_dim = h1 - h0, k1 - k0, d
        self.qkv = Linear(
            ckpt,
            [
                Source(f"{p}.q_proj", ((2 * d * h0, 2 * d * h1),)),
                Source(f"{p}.k_proj", ((d * k0, d * k1),)),
                Source(f"{p}.v_proj", ((d * k0, d * k1),)),
            ],
            dtype, device, keep_dense=keep_dense,
        )
        self.o_proj = Linear(ckpt, [Source(f"{p}.o_proj")], dtype, device, cols=(d * h0, d * h1),
                             keep_dense=keep_dense)
        self.q_norm = _norm_weight(ckpt, f"{p}.q_norm.weight", device, 1.0)
        self.k_norm = _norm_weight(ckpt, f"{p}.k_norm.weight", device, 1.0)
        self.eps = cfg.rms_norm_eps
        self.scale = 1.0 / math.sqrt(d)
        self.par = par
        self.section = cfg.mrope_section

    def __call__(self, x, step: StepInput, cache: ModelCache, layer: int, rope, slots):
        t, d = x.shape[0], self.head_dim
        qkv = self.qkv(x)
        if qkv.is_cuda and d == 256:
            ks, vs = cache.k_scale.get(layer), cache.v_scale.get(layer)
            q = kernels.attn_prep(qkv, self.q_norm, self.k_norm, step.positions, self.inv_freq, self.section,
                                  step.kv_slots_t, cache.k[layer], cache.v[layer], self.heads, self.kv_heads, self.eps,
                                  ks, vs)
            if (self.heads // self.kv_heads) * step.max_query_len <= 64:
                o = kernels.decode_attention(q, qkv, cache.k[layer], cache.v[layer], step.block_tables,
                                             step.query_start_t, step.context_lens_t, step.max_query_len,
                                             step.max_blocks, self.scale, ks, vs)
            else:
                o = kernels.flash_attention(q, qkv, cache.k[layer], cache.v[layer], step.block_tables,
                                            step.query_start, step.query_start_t, step.context_lens_t, self.scale,
                                            ks, vs)
            return self.o_proj(o)
        qg = qkv[:, : 2 * d * self.heads].view(t, self.heads, 2, d)
        q = qg[:, :, 0].contiguous()
        gate = qg[:, :, 1].reshape(t, self.heads * d)
        k = qkv[:, 2 * d * self.heads : 2 * d * self.heads + d * self.kv_heads].contiguous()
        v = qkv[:, 2 * d * self.heads + d * self.kv_heads :].reshape(t, self.kv_heads, d)
        q = ops.rms_norm(q.view(-1, d), self.q_norm, self.eps)[0].view(t, self.heads, d)
        k = ops.rms_norm(k.view(-1, d), self.k_norm, self.eps)[0].view(t, self.kv_heads, d)
        cos, sin = rope
        q = ops.apply_rotary(q, cos, sin)
        k = ops.apply_rotary(k, cos, sin)
        ks, vs = cache.k_scale.get(layer), cache.v_scale.get(layer)
        ops.write_kv(cache.k[layer], cache.v[layer], k, v, slots, ks, vs)
        o = ops.paged_attention(q, cache.k[layer], cache.v[layer], step.query_start, step.context_lens,
                                step.block_tables, self.scale, ks, vs)
        o = ops.sigmoid_gate(o.reshape(t, self.heads * d), gate)
        return self.o_proj(o)


class GatedDeltaNetLayer:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, par: Parallel, dtype, device, keep_dense):
        dk, dv = cfg.gdn_k_dim, cfg.gdn_v_dim
        a0, a1 = par.span(cfg.gdn_k_heads)
        b0, b1 = par.span(cfg.gdn_v_heads)
        ks, vs = cfg.gdn_key_size, cfg.gdn_value_size
        self.hk, self.hv, self.dk, self.dv = a1 - a0, b1 - b0, dk, dv
        qkv_rows = ((a0 * dk, a1 * dk), (ks + a0 * dk, ks + a1 * dk), (2 * ks + b0 * dv, 2 * ks + b1 * dv))
        self.in_proj = Linear(
            ckpt,
            [
                Source(f"{p}.in_proj_qkv", qkv_rows),
                Source(f"{p}.in_proj_z", ((b0 * dv, b1 * dv),)),
                Source(f"{p}.in_proj_b", ((b0, b1),)),
                Source(f"{p}.in_proj_a", ((b0, b1),)),
            ],
            dtype, device, keep_dense=keep_dense,
        )
        self.out_proj = Linear(ckpt, [Source(f"{p}.out_proj")], dtype, device, cols=(b0 * dv, b1 * dv),
                               keep_dense=keep_dense)
        conv = ckpt.tensor(f"{p}.conv1d.weight").squeeze(1)
        self.conv_weight = torch.cat([conv[r0:r1] for r0, r1 in qkv_rows]).float().to(device)
        self.a_log = ckpt.tensor(f"{p}.A_log")[b0:b1].float().to(device)
        self.dt_bias = ckpt.tensor(f"{p}.dt_bias")[b0:b1].float().to(device)
        self.norm = _norm_weight(ckpt, f"{p}.norm.weight", device, 0.0)
        self.eps = cfg.rms_norm_eps
        self.par = par

    def __call__(self, x, step: StepInput, cache: ModelCache, layer: int, rope, slots, stash=None):
        t = x.shape[0]
        key, val = self.hk * self.dk, self.hv * self.dv
        y = self.in_proj(x)
        if y.is_cuda and step.max_query_len <= RECURRENT_MAX_TOKENS:
            o = kernels.gdn_recurrent(y, self.conv_weight, cache.conv[layer], cache.state[layer], self.a_log,
                                      self.dt_bias, self.norm, step.query_start_t, step.context_lens_t,
                                      step.state_slots_t, self.hk, self.hv, self.eps, commit=stash is None)
            if stash is not None:
                stash[layer] = y
            return self.out_proj(o)
        conv_dim = 2 * key + val
        qkv = ops.causal_conv1d(y[:, :conv_dim].contiguous(), self.conv_weight, cache.conv[layer],
                                step.query_start, step.context_lens, step.state_slots)
        z = y[:, conv_dim : conv_dim + val]
        b = y[:, conv_dim + val : conv_dim + val + self.hv]
        a = y[:, conv_dim + val + self.hv :]
        q = qkv[:, :key].view(t, self.hk, self.dk)
        k = qkv[:, key : 2 * key].view(t, self.hk, self.dk)
        v = qkv[:, 2 * key :].view(t, self.hv, self.dv)
        g, beta = ops.gdn_gates(a, b, self.a_log, self.dt_bias)
        if v.is_cuda:
            o = gdn_prefill.gated_delta_rule(q, k, v, g, beta, cache.state[layer], step.query_start,
                                             step.query_start_t, step.context_lens_t, step.state_slots_t)
        else:
            o = ops.gated_delta_rule(q, k, v, g, beta, cache.state[layer], step.query_start, step.context_lens,
                                     step.state_slots)
        o = ops.gated_rms_norm(o.reshape(t * self.hv, self.dv), z.reshape(t * self.hv, self.dv), self.norm, self.eps)
        return self.out_proj(o.view(t, val))


class MLP:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, par: Parallel, dtype, device, keep_dense):
        i0, i1 = par.span(cfg.intermediate_size)
        self.gate_up = Linear(ckpt, [Source(f"{p}.gate_proj", ((i0, i1),)), Source(f"{p}.up_proj", ((i0, i1),))],
                              dtype, device, keep_dense=keep_dense)
        self.down = Linear(ckpt, [Source(f"{p}.down_proj")], dtype, device, cols=(i0, i1), keep_dense=keep_dense)
        self.par = par

    def __call__(self, x):
        return self.down(ops.silu_mul(self.gate_up(x)))


class Layer:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, index: int, par: Parallel, dtype, device, keep_dense):
        p = f"{ckpt.text_prefix}layers.{index}"
        self.kind = cfg.layer_types[index]
        if self.kind == "full_attention":
            self.mixer = AttentionLayer(cfg, ckpt, f"{p}.self_attn", par, dtype, device, keep_dense)
        else:
            self.mixer = GatedDeltaNetLayer(cfg, ckpt, f"{p}.linear_attn", par, dtype, device, keep_dense)
        self.mlp = MLP(cfg, ckpt, f"{p}.mlp", par, dtype, device, keep_dense)
        self.input_norm = _norm_weight(ckpt, f"{p}.input_layernorm.weight", device, 1.0)
        self.post_norm = _norm_weight(ckpt, f"{p}.post_attention_layernorm.weight", device, 1.0)


class Qwen35TextModel:
    def __init__(
        self,
        cfg: ModelConfig,
        ckpt: Checkpoint,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        par: Parallel = SINGLE,
        layers: int | None = None,
        keep_dense: bool = False,
    ):
        t = cfg.text
        self.cfg, self.par, self.dtype, self.device = t, par, dtype, torch.device(device)
        p = ckpt.text_prefix
        v0, v1 = par.span(t.vocab_size)
        self.vocab_span = (v0, v1)
        self.embed = _to_dtype(ckpt.rows(f"{p}embed_tokens.weight", v0, v1), dtype).to(device)
        count = t.num_layers if layers is None else layers
        self.layers = [Layer(t, ckpt, i, par, dtype, device, keep_dense) for i in range(count)]
        half = t.rotary_dim // 2
        inv_freq = 1.0 / (t.rope_theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / t.rotary_dim))
        for layer in self.layers:
            if layer.kind == "full_attention":
                layer.mixer.inv_freq = inv_freq.float().to(device)
        self.final_norm = _norm_weight(ckpt, f"{p}norm.weight", device, 1.0)
        head = "lm_head.weight" if not t.tie_word_embeddings else f"{p}embed_tokens.weight"
        self.lm_head = _to_dtype(ckpt.rows(head, v0, v1), dtype).to(device)
        self.lm_head_packed = None
        if self.device.type == "cuda" and dtype == torch.float16:
            from ..ops.kernels import DenseWeight

            self.lm_head_packed = DenseWeight(self.lm_head)
            self.lm_head = None
            torch.cuda.empty_cache()
        half = t.rotary_dim // 2
        self.inv_freq = 1.0 / (t.rope_theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / t.rotary_dim))
        self.inv_freq = self.inv_freq.float().to(device)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        v0, v1 = self.vocab_span
        if self.par.size == 1:
            return self.embed[ids]
        inside = (ids >= v0) & (ids < v1)
        rows = self.embed[(ids - v0).clamp(0, v1 - v0 - 1)] * inside[:, None].to(self.dtype)
        return self.par.all_reduce(rows)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's FP32 logits [rows, vocab shard] of hidden states x."""
        from ..ops.kernels import dense_gemm

        x = x.contiguous()
        return torch.cat([dense_gemm(x[i : i + 64], self.lm_head_packed) for i in range(0, x.shape[0], 64)]) \
            if x.shape[0] else torch.empty(0, self.lm_head_packed.n, dtype=torch.float32, device=x.device)

    def local_logits(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's FP32 logits [rows, vocab shard] of final hidden states (FP16 products, FP32 accumulation)."""
        if self.lm_head_packed is not None:
            return self.head(x)
        out = torch.empty(x.shape[0], self.lm_head.shape[0], dtype=torch.float32, device=x.device)
        xf = x.float()
        for s in range(0, self.lm_head.shape[0], 32768):
            out[:, s : s + 32768] = xf @ self.lm_head[s : s + 32768].float().T
        return out

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 logits [rows, vocab] of final hidden states (FP16 products, FP32 accumulation and output)."""
        out = self.local_logits(x)
        if self.par.size == 1:
            return out
        parts = [torch.empty_like(out) for _ in range(self.par.size)]
        torch.distributed.all_gather(parts, out)
        return torch.cat(parts, dim=-1)

    def head_top(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(argmax, top k logits, their token ids) of final hidden states x, from each rank's vocabulary shard: only
        the shards' top k values and ids (one all-gather) cross the ranks, not the full logits."""
        local = self.local_logits(x)
        v, i = torch.topk(local, min(k, local.shape[-1]), dim=-1)
        i = (i + self.vocab_span[0]).int()
        if self.par.size == 1:
            return merge_top([v], [i], v.shape[-1])
        both = torch.cat((v, i.view(torch.float32)), dim=-1)
        parts = [torch.empty_like(both) for _ in range(self.par.size)]
        torch.distributed.all_gather(parts, both)
        kk = v.shape[-1]
        return merge_top([p[:, :kk] for p in parts], [p[:, kk:].view(torch.int32) for p in parts], kk)

    @torch.inference_mode()
    def forward(
        self, step: StepInput, cache: ModelCache, capture: tuple[int, ...] = (), stash: dict | None = None,
        top: int = 0,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run one step. Returns the logits of step.logit_rows (or, with top, head_top of those rows), and the
        output hidden states (all rows) of the layers listed in capture.

        With stash (a dict, speculative verify): Gated DeltaNet layers leave their state unchanged and keep their
        in_proj outputs in stash; commit() then writes the state for the accepted rows."""
        t = self.cfg
        h = self.embed_tokens(step.token_ids)
        for start, rows in step.embeds.items():
            h[start : start + rows.shape[0]] = rows.to(h.dtype)
        if h.is_cuda:
            rope = slots = None
        else:
            rope = ops.rotary_cos_sin(step.positions, self.inv_freq, t.mrope_section)
            slots = ops.kv_slots(step.query_start, step.context_lens, step.block_tables, step.block_size)
        residual = h
        x = ops.rms_norm(h, self.layers[0].input_norm, t.rms_norm_eps)[0]
        captured = {}
        for i, layer in enumerate(self.layers):
            if stash is not None and layer.kind != "full_attention":
                x = layer.mixer(x, step, cache, i, rope, slots, stash)
            else:
                x = layer.mixer(x, step, cache, i, rope, slots)
            x, residual = self.par.reduce_norm(x, layer.post_norm, t.rms_norm_eps, residual)
            x = layer.mlp(x)
            nxt = self.layers[i + 1].input_norm if i + 1 < len(self.layers) else self.final_norm
            x, residual = self.par.reduce_norm(x, nxt, t.rms_norm_eps, residual, full_residual=i in capture)
            if i in capture:
                captured[i] = residual.clone()
        if top:
            return self.head_top(x[step.logit_rows], top), captured
        return self.logits(x[step.logit_rows]), captured

    @torch.inference_mode()
    def commit(self, step: StepInput, cache: ModelCache, stash: dict, commit_lens: torch.Tensor) -> None:
        """After a speculative verify step: write each Gated DeltaNet layer's state as if only the first
        commit_lens[b] new tokens of every sequence had run."""
        for i, layer in enumerate(self.layers):
            if layer.kind == "full_attention":
                continue
            m = layer.mixer
            kernels.gdn_recurrent(stash[i], m.conv_weight, cache.conv[i], cache.state[i], m.a_log, m.dt_bias, m.norm,
                                  step.query_start_t, step.context_lens_t, step.state_slots_t, m.hk, m.hv, m.eps,
                                  commit_lens=commit_lens)
