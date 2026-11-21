"""DFlash2 drafter (DFlash2DraftModel checkpoints): drafts a block of tokens in one pass, then a candidate selector
traces one path through the top candidates of each position. It shares the target's embeddings and LM head.

Context: for every token the target has processed, the target's residual stream after its target layers is
concatenated, projected by fc and normalized; each drafter layer's key and value projections turn it into that
layer's keys (normalized, rotated at the token's position) and values. They live in a ring of R positions per
state slot (R covers the sliding window plus one block).

Block: [anchor, mask x (block - 1)] at positions P .. P + block - 1, where the anchor is the newest token (sampled
by the target, not yet processed by it). Each layer: RMS norm, a two-tap dynamic grouped convolution, attention
over the context in the window and the whole block (no causal mask in the block), a second convolution with the
other side's taps; the same around the MLP. Each mask position predicts the token at its own position.

Numerics: the drafter was trained in BF16, and its residual stream grows past the FP16 range. The residual stream,
the convolutions and every large intermediate stay in FP32; the tensor cores see FP16 inputs that fit (norm
outputs, attention outputs, and the MLP's down-projection input scaled per row), with FP32 outputs. On GPUs the
layers' weights, the input projection (fc), the context keys and values (kv_all), the selector's hidden
projection, the convolutions' kernel projections and the candidate head (a copy of the target's LM head shard) are
FP8 E4M3 with one scale per output row: half the weight traffic, and half the drafter's memory, which goes to the
KV cache. SEVENTY_DRAFTER_FP8=0 keeps them all FP16; SEVENTY_DRAFTER_FP8_EXTRA lists which of fc, kv_all,
sel_hidden and conv are FP8 (none: only the layers and the candidate head). Acceptance uses this drafter's own
draft probabilities, so the target's output distribution is unchanged whatever the drafter's precision. Measured
with sampling (thinking on, temperature 0.6, 2K and 32K prompts, 4 streams): 31.9% of the drafted tokens accepted
with FP16 layers and 32.0% with FP8 (two runs each; runs of one setting differ by about 3 points). Greedy, 4
streams: 2.47 / 2.38 accepted drafts per step and sequence at 2K / 32K with fc, kv_all, sel_hidden and conv in
FP16, 2.48 / 2.38 with them in FP8, 1.3 to 1.5% faster.

Selector: the top-k tokens of each mask position (from the LM head), and edge scores
score[l, p, c] = logit[l, c] + sum_r pred[token before (l, p)]_r * hidden[l]_r * succ[candidate (l, c)]_r,
where the token before position 0 is the anchor. A walk from the anchor picks one candidate per position
(argmax, or a draw at the request's temperature); the draw's distribution is the draft probability for
lossless rejection sampling.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import ops
from ..checkpoint import Checkpoint
from ..ops import reference as R
from ..parallel import SINGLE, Parallel
from .layers import _to_dtype


@dataclass(frozen=True)
class DrafterConfig:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    layers: int
    eps: float
    rope_theta: float
    window: int
    block: int
    mask_token: int
    taps: int
    group_size: int
    selector_rank: int
    top_k: int
    target_layers: tuple[int, ...]

    @staticmethod
    def load(path: str | Path) -> "DrafterConfig":
        raw = json.loads((Path(path) / "config.json").read_text())
        if "DFlash2DraftModel" not in (raw.get("architectures") or []):
            raise ValueError(f"{path} is not a DFlash2 drafter (architectures {raw.get('architectures')})")
        d = raw["dflash_config"]
        rope = raw.get("rope_parameters") or {}
        return DrafterConfig(
            hidden=raw["hidden_size"], heads=raw["num_attention_heads"], kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"],
            intermediate=raw["intermediate_size"], layers=raw["num_hidden_layers"], eps=raw["rms_norm_eps"],
            rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", 1e6))),
            window=int(raw.get("sliding_window") or 2048), block=int(d["block_size"]),
            mask_token=int(d["mask_token_id"]), taps=int(d["conv_kernel_size"]), group_size=int(d["conv_group_size"]),
            selector_rank=int(d["selector_rank"]), top_k=int(d["selector_top_k"]),
            target_layers=tuple(int(x) for x in d["target_layer_ids"]),
        )

    @property
    def ring(self) -> int:
        return -(-(self.window + self.block) // 16) * 16


@dataclass
class Draft:
    tokens: torch.Tensor
    probs: torch.Tensor
    candidates: torch.Tensor


class DraftLayer:
    def __init__(self, cfg: DrafterConfig, ckpt: Checkpoint, i: int, par: Parallel, dtype, device):
        p = f"layers.{i}"
        d = cfg.head_dim
        h0, h1 = par.span(cfg.heads)
        k0, k1 = par.span(cfg.kv_heads)
        self.heads, self.kv = h1 - h0, k1 - k0

        def w(name, rows=None, cols=None):
            return _to_dtype(ckpt.dense_part(f"{p}.{name}.weight", rows=rows, cols=cols), dtype).to(device)

        self.q = w("self_attn.q_proj", rows=(h0 * d, h1 * d))
        self.kv_w = torch.cat((w("self_attn.k_proj", rows=(k0 * d, k1 * d)), w("self_attn.v_proj", rows=(k0 * d, k1 * d))))
        self.o = w("self_attn.o_proj", cols=(h0 * d, h1 * d))
        i0, i1 = par.span(cfg.intermediate)
        self.gate_up = torch.cat((w("mlp.gate_proj", rows=(i0, i1)), w("mlp.up_proj", rows=(i0, i1))))
        self.down = w("mlp.down_proj", cols=(i0, i1))
        norm = lambda name: ckpt.tensor(f"{p}.{name}.weight").float().to(device)
        self.input_norm, self.post_norm = norm("input_layernorm"), norm("post_attention_layernorm")
        self.q_norm, self.k_norm = norm("self_attn.q_norm"), norm("self_attn.k_norm")
        self.conv = {}
        for side in ("attention_conv", "mlp_conv"):
            self.conv[side] = (ckpt.tensor(f"{p}.{side}.base_kernel").float().to(device),
                               _to_dtype(ckpt.tensor(f"{p}.{side}.kernel_projection.weight"), dtype).to(device))


class Drafter:
    def __init__(self, path: str | Path, target, device, dtype=torch.float16, par: Parallel = SINGLE,
                 num_slots: int = 1):
        self.cfg = cfg = DrafterConfig.load(path)
        self.target, self.par, self.device, self.dtype = target, par, torch.device(device), dtype
        self.fp8 = (self.device.type == "cuda" and dtype == torch.float16
                    and os.environ.get("SEVENTY_DRAFTER_FP8", "1") != "0")
        with Checkpoint(path) as ckpt:
            self.fc = _to_dtype(ckpt.tensor("fc.weight"), dtype).to(device)
            self.hidden_norm = ckpt.tensor("hidden_norm.weight").float().to(device)
            self.norm = ckpt.tensor("norm.weight").float().to(device)
            self.layers = [DraftLayer(cfg, ckpt, i, par, dtype, device) for i in range(cfg.layers)]
            self.sel_hidden = _to_dtype(ckpt.tensor("candidate_selector.hidden_projection.weight"), dtype).to(device)
            self.pred = _to_dtype(ckpt.tensor("candidate_selector.predecessor_codebook"), dtype).to(device)
            self.succ = _to_dtype(ckpt.tensor("candidate_selector.successor_codebook"), dtype).to(device)
        self.kv_all = torch.cat([layer.kv_w for layer in self.layers])
        self.cand_head = None
        self.fused = (self.device.type == "cuda" and dtype == torch.float16 and cfg.head_dim == 128
                      and os.environ.get("SEVENTY_DRAFTER_FUSED", "1") != "0")
        if self.device.type == "cuda" and dtype == torch.float16:
            from ..ops.kernels import DenseWeight, DenseWeightFP8

            fp8_extra = set(os.environ.get("SEVENTY_DRAFTER_FP8_EXTRA", "fc,kv_all,sel_hidden,conv").split(","))

            for layer in self.layers:
                if self.fused:
                    layer.q_cols = layer.q.shape[0]
                    layer.qkv_src_q, layer.qkv_src_kv = layer.q, layer.kv_w
                    layer.qkv = DenseWeight(torch.cat((layer.q, layer.kv_w)))
                    layer.q = layer.kv_w = None
                lossy = self.fp8
                if lossy and self.fused:
                    layer.qkv = DenseWeightFP8(torch.cat((layer.qkv_src_q, layer.qkv_src_kv)))
                for name in ("q", "kv_w", "o", "gate_up", "down"):
                    if getattr(layer, name) is not None:
                        setattr(layer, name, (DenseWeightFP8 if lossy else DenseWeight)(getattr(layer, name)))
                layer.qkv_src_q = layer.qkv_src_kv = None
                conv_w = DenseWeightFP8 if lossy and "conv" in fp8_extra else DenseWeight
                layer.conv = {side: (base, conv_w(proj)) for side, (base, proj) in layer.conv.items()}
            pick = lambda name: DenseWeightFP8 if self.fp8 and name in fp8_extra else DenseWeight
            self.fc, self.kv_all = pick("fc")(self.fc), pick("kv_all")(self.kv_all)
            self.sel_hidden = pick("sel_hidden")(self.sel_hidden)
            if self.fp8 and target.lm_head_packed is not None:
                self.cand_head = DenseWeightFP8.from_dense(target.lm_head_packed)
            torch.cuda.empty_cache()
        self.k_ring = self.v_ring = None
        if num_slots:
            self.allocate(num_slots)
        half = cfg.head_dim // 2
        self.inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, half, dtype=torch.float64) * 2 / cfg.head_dim))
                         ).float().to(device)
        self.groups = cfg.hidden // cfg.group_size

    def ring_bytes_per_slot(self) -> int:
        cfg = self.cfg
        return 2 * cfg.layers * cfg.ring * self.layers[0].kv * cfg.head_dim * torch.finfo(self.dtype).bits // 8

    def allocate(self, num_slots: int) -> None:
        """The context rings for num_slots state slots (plus the scratch slot)."""
        cfg = self.cfg
        shape = (cfg.layers, num_slots + 1, cfg.ring, self.layers[0].kv, cfg.head_dim)
        self.k_ring = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_ring = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ang = positions.float()[:, None] * self.inv_freq
        return torch.cos(ang), torch.sin(ang)

    def _norm(self, x, weight, residual=None):
        """RMS norm in FP32; returns (normalized FP16, new FP32 residual or None)."""
        if self.fused:
            from ..ops import kernels

            return kernels.drafter_norm(x, weight, self.cfg.eps, residual)
        s = x.float() if residual is None else residual + x.float()
        out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + self.cfg.eps) * weight
        return out.to(self.dtype), (None if residual is None else s)

    @staticmethod
    def _mm(x: torch.Tensor, w, row_scale: torch.Tensor | None = None) -> torch.Tensor:
        """x (FP16) @ w^T with FP32 output (times row_scale per row, packed weights only). Packed weights use the
        small-batch FP16 GEMM, 64 rows at a time."""
        from ..ops.kernels import DenseWeight, DenseWeightFP8, dense_gemm

        if isinstance(w, (DenseWeight, DenseWeightFP8)):
            x = x.contiguous()
            if x.shape[0] <= 64:
                return dense_gemm(x, w, row_scale=row_scale)
            return torch.cat([dense_gemm(x[i : i + 64], w, row_scale=None if row_scale is None else row_scale[i : i + 64])
                              for i in range(0, x.shape[0], 64)])
        return torch.mm(x, w.T, out_dtype=torch.float32) if x.is_cuda else x.float() @ w.float().T

    def _mm_scaled(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x (FP32, any range) @ w^T: rows are scaled into the FP16 range for the tensor cores and back."""
        scale = (x.abs().amax(-1, keepdim=True) / 16384.0).clamp(min=1.0)
        return self._mm((x / scale).to(self.dtype), w) * scale

    def copy_slot(self, src: int, dst: int) -> None:
        self.k_ring[:, dst].copy_(self.k_ring[:, src])
        self.v_ring[:, dst].copy_(self.v_ring[:, src])

    def write_context(self, features: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor) -> None:
        """features: [N, layers * hidden] (the target's residual stream after its target layers, concatenated);
        positions and slots: [N]. Writes the keys and values of every drafter layer to the ring."""
        if features.shape[0] == 0:
            return
        cfg = self.cfg
        ctx = self._norm(self._mm(features.to(self.dtype), self.fc), self.hidden_norm)[0]
        kv = self.layers[0].kv
        d = cfg.head_dim
        if self.fused:
            from ..ops import kernels

            flat = self._mm(ctx, self.kv_all)
            width = 2 * kv * d
            for li, layer in enumerate(self.layers):
                kernels.drafter_qk(None, flat[:, li * width : (li + 1) * width], None, layer.k_norm, positions, slots,
                                   self.inv_freq, self.k_ring[li], self.v_ring[li], 0, kv, cfg.eps)
            return
        allkv = self._mm(ctx, self.kv_all).view(-1, cfg.layers, 2, kv, d)
        cos, sin = self._rope(positions)
        idx = positions % cfg.ring
        for li, layer in enumerate(self.layers):
            k = self._norm(allkv[:, li, 0].reshape(-1, d), layer.k_norm)[0].view(-1, kv, d)
            self.k_ring[li, slots, idx] = R.apply_rotary(k, cos, sin)
            self.v_ring[li, slots, idx] = allkv[:, li, 1].to(self.dtype)

    def _conv(self, x: torch.Tensor, base: torch.Tensor, delta: torch.Tensor, out_dtype=None) -> torch.Tensor:
        """Two-tap grouped convolution within each block of `block` rows: out[t] = c0[t] x[t] + c1[t] x[t - 1]."""
        cfg = self.cfg
        if self.fused:
            from ..ops import kernels

            return kernels.drafter_conv(x, base, delta, cfg.group_size, cfg.block,
                                        (out_dtype or x.dtype) == torch.float32)
        t = x.shape[0]
        xs = x.float().view(t, self.groups, cfg.group_size)
        coef = base.view(1, cfg.taps, self.groups, cfg.group_size) + delta.float().unsqueeze(-1)
        out = coef[:, 0] * xs
        pos = torch.arange(t, device=x.device) % cfg.block
        for tap in range(1, cfg.taps):
            shifted = F.pad(xs[:-tap], (0, 0, 0, 0, tap, 0))
            out = out + coef[:, tap] * shifted * (pos >= tap).view(-1, 1, 1)
        return out.view(t, -1).to(out_dtype or x.dtype)

    def _attention(self, layer: DraftLayer, li: int, h: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                   ends: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b = slots.numel()
        blk, d = cfg.block, cfg.head_dim
        if self.fused:
            from ..ops import kernels

            qkv = self._mm(h, layer.qkv)
            q16 = kernels.drafter_qk(qkv[:, : layer.q_cols], qkv[:, layer.q_cols :], layer.q_norm, layer.k_norm,
                                     positions, slots.repeat_interleave(blk), self.inv_freq, self.k_ring[li],
                                     self.v_ring[li], layer.heads, layer.kv, cfg.eps)
            o = kernels.drafter_attention(q16, self.k_ring[li], self.v_ring[li], slots, ends - blk, blk, cfg.window,
                                          1.0 / math.sqrt(d))
            return self.par.all_reduce(self._mm(o, layer.o))
        q = self._mm(h, layer.q).view(-1, layer.heads, d)
        kv = self._mm(h, layer.kv_w).view(-1, 2, layer.kv, d)
        q = self._norm(q.reshape(-1, d), layer.q_norm)[0].view(-1, layer.heads, d)
        k = self._norm(kv[:, 0].reshape(-1, d), layer.k_norm)[0].view(-1, layer.kv, d)
        cos, sin = self._rope(positions)
        q, k = R.apply_rotary(q, cos, sin), R.apply_rotary(k, cos, sin)
        ring_slots = slots.repeat_interleave(blk)
        self.k_ring[li, ring_slots, positions % cfg.ring] = k
        self.v_ring[li, ring_slots, positions % cfg.ring] = kv[:, 1].to(self.dtype)
        keys = self.k_ring[li, slots].float()
        vals = self.v_ring[li, slots].float()
        ring = torch.arange(cfg.ring, device=h.device)
        last = ends[:, None] - 1
        kpos = last - ((last - ring[None, :]) % cfg.ring)
        qpos = positions.view(b, blk)
        visible = (kpos[:, None, :] >= 0) & (kpos[:, None, :] >= qpos[:, :, None] - (cfg.window - 1))
        rep = layer.heads // layer.kv
        qq = q.float().view(b, blk, layer.kv, rep, d)
        scores = torch.einsum("btgrd,bsgd->bgrts", qq, keys) / math.sqrt(d)
        scores = scores.masked_fill(~visible[:, None, None], float("-inf"))
        o = torch.einsum("bgrts,bsgd->btgrd", torch.softmax(scores, dim=-1), vals)
        o = o.reshape(b * blk, layer.heads * d)
        return self.par.all_reduce(self._mm_scaled(o, layer.o))

    def hidden(self, anchors: torch.Tensor, starts: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Final hidden states [B * block, hidden] of the blocks [anchor, mask ...] at positions start .. start + block."""
        cfg = self.cfg
        b, blk = anchors.numel(), cfg.block
        ids = torch.full((b, blk), cfg.mask_token, dtype=torch.long, device=anchors.device)
        ids[:, 0] = anchors
        positions = (starts[:, None] + torch.arange(blk, device=anchors.device)).reshape(-1)
        ends = starts + blk
        x = self.target.embed_tokens(ids.reshape(-1)).float()
        residual = None
        for li, layer in enumerate(self.layers):
            if residual is None:
                h, residual = self._norm(x, layer.input_norm)[0], x
            else:
                h, residual = self._norm(x, layer.input_norm, residual)
            base, proj = layer.conv["attention_conv"]
            coef = self._mm(h, proj).view(-1, 2, cfg.taps, self.groups)
            a = self._attention(layer, li, self._conv(h, base[0], coef[:, 0]), positions, slots, ends)
            a = self._conv(a, base[1], coef[:, 1], torch.float32)
            h, residual = self._norm(a, layer.post_norm, residual)
            base, proj = layer.conv["mlp_conv"]
            coef = self._mm(h, proj).view(-1, 2, cfg.taps, self.groups)
            gu = self._mm(self._conv(h, base[0], coef[:, 0]), layer.gate_up)
            if self.fused:
                from ..ops import kernels

                p16, sc = kernels.silu_mul_scale(gu)
                m = self._mm(p16, layer.down, row_scale=sc)
            else:
                gate, up = gu.chunk(2, dim=-1)
                m = self._mm_scaled(F.silu(gate) * up, layer.down)
            x = self._conv(self.par.all_reduce(m), base[1], coef[:, 1], torch.float32)
        return self._norm(x, self.norm, residual)[0]

    def _candidates(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Global top-k (values, token ids) of the target LM head for rows of h."""
        k = self.cfg.top_k
        if self.cand_head is not None:
            from ..ops.kernels import dense_gemm

            h = h.contiguous()
            logits = torch.cat([dense_gemm(h[i : i + 64], self.cand_head) for i in range(0, h.shape[0], 64)])
        elif self.target.lm_head_packed is not None:
            logits = self.target.head(h)
        else:
            logits = h.float() @ self.target.lm_head.float().T
        if not logits.is_cuda or k > 64 or logits.dtype != torch.float32:
            values, ids = torch.topk(logits, k, dim=-1)
            ids = ids + self.target.vocab_span[0]
            if self.par.size > 1:
                vs = [torch.empty_like(values) for _ in range(self.par.size)]
                ks = [torch.empty_like(ids) for _ in range(self.par.size)]
                torch.distributed.all_gather(vs, values)
                torch.distributed.all_gather(ks, ids)
                values, sel = torch.topk(torch.cat(vs, -1), k, dim=-1)
                ids = torch.cat(ks, -1).gather(-1, sel)
            return values, ids
        from ..ops import kernels

        values, ids = kernels.topk(logits.contiguous(), k, id_offset=self.target.vocab_span[0])
        if self.par.size > 1:
            both = torch.cat((values, ids.view(torch.float32)), dim=-1)
            parts = [torch.empty_like(both) for _ in range(self.par.size)]
            torch.distributed.all_gather(parts, both)
            values, ids = kernels.topk(torch.cat([p[:, :k] for p in parts], dim=-1), k,
                                       ids=torch.cat([p[:, k:].view(torch.int32) for p in parts], dim=-1))
        return values, ids.long()

    def walk(self, anchors, values, ids, h, temps: torch.Tensor, gumbel: torch.Tensor | None) -> Draft:
        """The selector walk for a batch, on the device. gumbel: [B, positions, top_k] noise (or None: greedy)."""
        cfg = self.cfg
        b, n, kk = ids.shape
        hid = self._mm(h, self.sel_hidden).view(b, n, -1)
        if (ids.is_cuda and kk <= 32 and self.pred.dtype == torch.float16 and self.pred.shape[1] % 32 == 0
                and self.pred.shape[1] <= 1024 and os.environ.get("SEVENTY_DRAFTER_WALK") != "torch"):
            from ..ops import kernels

            tokens, probs = kernels.drafter_walk(anchors.view(b), values, ids, hid, self.pred, self.succ, temps,
                                                 gumbel)
            return Draft(tokens=tokens, probs=probs, candidates=ids)
        prev = torch.cat((anchors.view(b, 1, 1).expand(b, 1, kk), ids[:, :-1]), dim=1)
        edges = values[:, :, None, :] + torch.einsum("blpr,blcr->blpc", self.pred[prev].float() * hid[:, :, None],
                                                      self.succ[ids].float())
        rows = torch.arange(b, device=ids.device)
        idx = torch.zeros(b, dtype=torch.long, device=ids.device)
        greedy = temps == 0
        safe_t = torch.where(greedy, torch.ones_like(temps), temps)
        tokens, probs = [], []
        for step in range(n):
            scores = edges[rows, step, idx]
            scaled = scores / safe_t[:, None]
            noisy = scaled if gumbel is None else torch.where(greedy[:, None], scaled, scaled + gumbel[:, step])
            choice = noisy.argmax(-1)
            p = torch.where(greedy[:, None], F.one_hot(choice, kk).float(), torch.softmax(scaled, dim=-1))
            tokens.append(ids[rows, step, choice])
            probs.append(p)
            idx = choice
        return Draft(tokens=torch.stack(tokens, 1), probs=torch.stack(probs, 1), candidates=ids)

    def draft(self, anchors: torch.Tensor, starts: torch.Tensor, slots: torch.Tensor, temperatures: list[float],
              generators: list) -> Draft:
        """block - 1 draft tokens per sequence, with their draft probabilities over the candidates."""
        cfg = self.cfg
        b, n = anchors.numel(), cfg.block - 1
        h = self.hidden(anchors, starts, slots).view(b, cfg.block, -1)[:, 1:].reshape(b * n, -1)
        values, ids = self._candidates(h)
        values, ids = values.view(b, n, -1), ids.view(b, n, -1)
        hid = self._mm(h, self.sel_hidden).view(b, n, -1)
        prev = torch.cat((anchors.view(b, 1, 1).expand(b, 1, cfg.top_k), ids[:, :-1]), dim=1)
        pred = self.pred[prev].float()
        succ = self.succ[ids].float()
        edges = values[:, :, None, :] + torch.einsum("blpr,blcr->blpc", pred * hid[:, :, None], succ)
        tokens = torch.empty(b, n, dtype=torch.long, device=anchors.device)
        probs = torch.empty(b, n, cfg.top_k, dtype=torch.float32, device=anchors.device)
        idx = torch.zeros(b, dtype=torch.long, device=anchors.device)
        rows = torch.arange(b, device=anchors.device)
        temps = torch.tensor([max(t, 0.0) for t in temperatures], device=anchors.device)
        for step in range(n):
            scores = edges[rows, step, idx]
            greedy = temps == 0
            scaled = scores / torch.where(greedy, torch.ones_like(temps), temps)[:, None]
            p = torch.softmax(scaled, dim=-1)
            choice = scores.argmax(-1)
            for i in torch.nonzero(~greedy).flatten().tolist():
                choice[i] = torch.multinomial(p[i].cpu(), 1, generator=generators[i])[0]
            probs[:, step] = torch.where(greedy[:, None], F.one_hot(choice, cfg.top_k).float(), p)
            tokens[:, step] = ids[rows, step, choice]
            idx = choice
        return Draft(tokens=tokens, probs=probs, candidates=ids)
