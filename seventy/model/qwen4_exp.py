"""The Qwen4-Exp text model (Qwen3.8 Flash-Next): the Qwen3.5 hybrid of Gated DeltaNet and gated full-attention
layers with three additions (config.ExpConfig).

- Hyper-connections. The residual is hc_count streams of hidden_size ([tokens, hc_count x hidden]). Before each
  attention and MLP block, a grouped RMS norm of the streams (one norm per stream) feeds a low-rank gate
  (down, silu(. / hc_count), up, sigmoid) that weighs each stream; the block input is the mean of the weighted normed
  streams. The block output adds to every stream, scaled by 2 sigmoid(inject(normed) / hc_count) per stream. After
  the last layer, the same mix without injection gives the LM head's input. There are no other norms around the
  blocks.
- Per-layer n-gram embeddings (PLE) at ple_layers (layer 1 in Flash-Next): see NGramTable and PLE.
- Qwen Sparse Attention: see Indexer. A query attends to all tokens while the context has no more than
  indexer_budget tokens in whole blocks; past that, to the tokens of its best indexer_budget / indexer_compress
  blocks and of the unfinished last block.

The Gated DeltaNet layers gate their output norm with a sigmoid (Qwen3.5: silu). The MLP of every layer is a mixture
of experts (moe.MoE).

Tensor parallel: attention, Gated DeltaNet and the MoE split as in Qwen3.5; the hyper-connection and PLE weights
are on every GPU, and the n-gram table rows are read from host memory. Every block output is complete on every rank
after its all-reduce, so each rank keeps the full residual streams.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .. import nvfp4, ops
from ..checkpoint import Checkpoint
from ..config import ModelConfig, TextConfig
from ..engine.cache import ModelCache
from ..engine.step import StepInput
from ..parallel import SINGLE, VOCAB_ALIGN, Parallel
from ..rope import frequencies
from ..upload import to_device
from .layers import Linear, Source, _to_dtype
from ..split import balanced
from .moe import MoE
from .qwen3_5 import FINAL, GDN_COMMIT_LAYERS, AttentionLayer, GatedDeltaNetLayer, Qwen35TextModel


def _grouped_norm_weight(ckpt: Checkpoint, name: str, device) -> torch.Tensor:
    return (ckpt.tensor(name).float() + 1.0).to(device)


def grouped_rms_norm(x: torch.Tensor, weight: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    """RMS norm of each group of `group` values of x [T, n], times weight [n] (already 1 + w), in FP32."""
    f = x.float().view(x.shape[0], -1, group)
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return f.view(x.shape[0], -1) * weight


class HyperConnection:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, dtype, device, combine: bool = True):
        e = cfg.exp
        self.hc, self.hidden, self.eps = e.hc_count, cfg.hidden_size, cfg.rms_norm_eps
        self.norm = _grouped_norm_weight(ckpt, f"{p}.hc_norm.weight", device)
        self.down = Linear(ckpt, [Source(f"{p}.input_mix_weight_down")], torch.float32, device, keep_dense=True)
        self.up = Linear(ckpt, [Source(f"{p}.input_mix_weight_up")], torch.float32, device, keep_dense=True)
        self.inject = (Linear(ckpt, [Source(f"{p}.block_inject_weight")], torch.float32, device, keep_dense=True)
                       if combine else None)
        self.rank = self.down.dense.shape[0]
        self.packed = None
        if torch.device(device).type == "cuda" and dtype == torch.float16:
            from ..ops.kernels import DenseWeight

            first = torch.cat([self.down.dense] + ([self.inject.dense] if combine else []))
            self.packed = (DenseWeight(first.half()), DenseWeight(self.up.dense.half()))
            self.down.dense = self.up.dense = None
            if self.inject is not None:
                self.inject.dense = None
            torch.cuda.empty_cache()

    def __call__(self, h: torch.Tensor):
        """(block input [T, hidden] (FP16 on the GPU, FP32 on the CPU), injection weights [T, hc] FP32 or None) of the
        streams h [T, hc x hidden] FP32. On the GPU: seventy_hc_* kernels around two products with FP16 weights and
        FP32 sums (the dense GEMM kernel up to 64 rows, else cuBLAS tensor cores)."""
        if self.packed is not None and 0 < h.shape[0] <= 64:
            from ..ops import kernels

            first, up = self.packed
            n = kernels.hc_norm(h, self.norm, self.hc, self.eps)
            d = kernels.dense_gemm(n, first)
            s, inject = kernels.hc_act(d, self.rank, self.hc, self.inject is not None)
            u = kernels.dense_gemm(s, up)
            return kernels.hc_combine(u, n, self.hc), inject
        if self.packed is not None:
            from ..ops import kernels

            first, up = self.packed
            n = kernels.hc_norm(h, self.norm, self.hc, self.eps)
            d = torch.mm(n, kernels.dense_unpack(first).T, out_dtype=torch.float32)
            s, inject = kernels.hc_act(d, self.rank, self.hc, self.inject is not None)
            u = torch.mm(s, kernels.dense_unpack(up).T, out_dtype=torch.float32)
            return kernels.hc_combine(u, n, self.hc), inject
        n = grouped_rms_norm(h, self.norm, self.hidden, self.eps)
        mix = torch.sigmoid(F.linear(F.silu(F.linear(n, self.down.dense) / self.hc), self.up.dense))
        x = (mix * n).view(-1, self.hc, self.hidden).mean(dim=1)
        if self.inject is None:
            return x, None
        return x, 2.0 * torch.sigmoid(F.linear(n, self.inject.dense) / self.hc)


class NGramTable:
    """The n-gram embedding table of one PLE layer, read where the checkpoint stores it (memory-mapped shards, no
    copy): NVFP4 rows (packed values, E4M3 block scales of 16, one global scale) or dense rows (BF16, FP16, or FP8
    with one scale). rows(ids) dequantizes the rows of ids [n] to FP32 [n, dim]."""

    def __init__(self, ckpt: Checkpoint, p: str):
        names = [n for n in ckpt.names(f"{p}.shard_") if n.endswith(".weight")]
        names.sort(key=lambda n: int(n[len(f"{p}.shard_") :].split(".")[0]))
        self.shards, self.starts = [], [0]
        self.nvfp4 = f"{p}.weight_scale_2" in ckpt
        for name in names:
            values = _memmap(ckpt, name)
            scales = _memmap(ckpt, name + "_scale") if self.nvfp4 else None
            self.shards.append((values, scales))
            self.starts.append(self.starts[-1] + values.shape[0])
        self.starts_np = np.array(self.starts[:-1], dtype=np.int64)
        if self.nvfp4:
            self.dim = self.shards[0][0].shape[1] * 2
            self.global_scale = 1.0 / float(ckpt.tensor(f"{p}.weight_scale_2").float().reshape(-1)[0])
        else:
            self.dim = self.shards[0][0].shape[1]
            self.scale = float(ckpt.tensor(f"{p}.weight_scale").float().reshape(-1)[0]) if f"{p}.weight_scale" in ckpt else 1.0
            self.dtype = ckpt._slice(names[0]).get_dtype()

    @property
    def num_rows(self) -> int:
        return self.starts[-1]

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy().astype(np.int64)
        shard = np.searchsorted(self.starts_np, flat, side="right") - 1
        out = np.empty((flat.size, self.shards[0][0].shape[1]), dtype=self.shards[0][0].dtype)
        sc = np.empty((flat.size, self.shards[0][1].shape[1]), dtype=np.uint8) if self.nvfp4 else None
        for s in np.unique(shard):
            sel = np.nonzero(shard == s)[0]
            local = flat[sel] - self.starts[s]
            out[sel] = self.shards[s][0][local]
            if self.nvfp4:
                sc[sel] = self.shards[s][1][local]
        if self.nvfp4:
            w = nvfp4.Nvfp4Weight(torch.from_numpy(out), torch.from_numpy(sc), self.global_scale)
            return nvfp4.dequantize(w, torch.float32).to(ids.device)
        t = torch.from_numpy(out)
        if self.dtype == "BF16":
            t = t.view(torch.bfloat16)
        elif self.dtype == "F16":
            t = t.view(torch.float16)
        elif self.dtype == "F8_E4M3":
            t = t.view(torch.float8_e4m3fn)
        return (t.float() * self.scale).to(ids.device)


_NP_DTYPES = {"U8": np.uint8, "F8_E4M3": np.uint8, "BF16": np.uint16, "F16": np.uint16, "F32": np.float32}


def _memmap(ckpt: Checkpoint, name: str) -> np.ndarray:
    """A read-only memory map of one 2-D tensor in its safetensors file."""
    import json
    import struct

    file = ckpt._files[name]
    with open(file, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
    a, b = meta["data_offsets"]
    rows, cols = meta["shape"]
    return np.memmap(file, dtype=_NP_DTYPES[meta["dtype"]], mode="r", offset=8 + n + a, shape=(rows, cols))


class PLE:
    """Per-layer n-gram embeddings of one layer. For the token at position p and its predecessors (a token before an
    end-of-sequence token, or before the start, counts as the end-of-sequence token), each n-gram size n = 2 ..
    ngram_size mixes the last n tokens (product with a per-position odd multiplier, XOR over the positions, int64),
    and each of heads_per_ngram heads takes a row of the table (mix modulo the head's prime, plus the head's
    offset). The rows (ple_embed_dim values in all) give a key per stream and a shared value; a gate from each
    stream's normed activations and its normed key weighs the value; the weighted values, plus a dilated depthwise
    causal convolution (dilation ngram_size, with SiLU) of their norm, add to the streams. The state of a sequence
    is its last ngram_size - 1 tokens and the last (kernel - 1) x dilation convolution inputs."""

    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, device, dtype, eos: int, par: Parallel = SINGLE,
                 last: bool = True):
        e = cfg.exp
        self.hc, self.hidden, self.eps, self.eos = e.hc_count, cfg.hidden_size, cfg.rms_norm_eps, eos
        self.n = e.ngram_size
        self.heads_per_ngram = e.heads_per_ngram
        pe = f"{p}.ple_embedding"
        self.multipliers = ckpt.tensor(f"{pe}.layer_multipliers").long()
        self.vocab_sizes = ckpt.tensor(f"{pe}.ngram_heads_vocab_sizes").long()
        self.offsets = ckpt.tensor(f"{pe}.ngram_heads_offsets").long()
        self.table = NGramTable(ckpt, f"{pe}.ngram_embedding")
        self._global_scale = 1.0 / self.table.global_scale if self.table.nvfp4 else 1.0
        self.key_proj = Linear(ckpt, [Source(f"{p}.key_proj")], torch.float32, device, keep_dense=True)
        self.value_proj = Linear(ckpt, [Source(f"{p}.value_proj")], torch.float32, device, keep_dense=True)
        self.norm_key = _grouped_norm_weight(ckpt, f"{p}.norm_key.weight", device)
        self.norm_query = _grouped_norm_weight(ckpt, f"{p}.norm_query.weight", device)
        self.norm_conv = _grouped_norm_weight(ckpt, f"{p}.norm_conv.weight", device)
        self.conv = ckpt.tensor(f"{p}.conv1d.weight").float().squeeze(1).to(device)
        self.taps = self.conv.shape[1]
        self.dilation = self.n
        self.history = (self.taps - 1) * self.dilation
        self.par, self.last = par, last
        self.gpu = None
        if torch.device(device).type == "cuda" and self.table.nvfp4:
            self._to_gpu(ckpt, p, device, dtype)

    def _to_gpu(self, ckpt: Checkpoint, p: str, device, dtype) -> None:
        """The GPU path: this rank's heads (split.balanced) of the table in pinned host memory, which the gather
        kernel reads directly. The ranks sum their heads' columns of the embedding (an all-gather through the
        all-reduce) and each computes the key and value projections in full."""
        heads = self.vocab_sizes.numel()
        h0, h1 = balanced(heads, self.par.size, self.par.rank)
        dim = self.table.dim // 1
        head_dim = self.key_proj.dense.shape[1] // heads
        r0 = int(self.offsets[h0])
        r1 = int(self.offsets[h1 - 1] + self.vocab_sizes[h1 - 1])
        packed = torch.empty(r1 - r0, dim // 2, dtype=torch.uint8, pin_memory=True)
        scales = torch.empty(r1 - r0, dim // 16, dtype=torch.uint8, pin_memory=True)
        t = self.table
        for s, (values, sc) in enumerate(t.shards):
            a, b = max(r0, t.starts[s]), min(r1, t.starts[s + 1])
            if a >= b:
                continue
            packed[a - r0 : b - r0] = torch.from_numpy(np.ascontiguousarray(values[a - t.starts[s] : b - t.starts[s]]))
            scales[a - r0 : b - r0] = torch.from_numpy(np.ascontiguousarray(sc[a - t.starts[s] : b - t.starts[s]]))
        key = Linear(ckpt, [Source(f"{p}.key_proj")], torch.float16, device)
        value = Linear(ckpt, [Source(f"{p}.value_proj")], torch.float16, device)
        self.gpu = dict(h0=h0, h1=h1, row_base=r0, packed=packed, scales=scales, key=key, value=value,
                        mult=self.multipliers.to(device), sizes=self.vocab_sizes.to(device),
                        offsets=self.offsets.to(device), head_dim=head_dim)
        self.key_proj = self.value_proj = None
        self.table = None
        torch.cuda.empty_cache()

    def _gpu_call(self, h: torch.Tensor, step: StepInput, cache: ModelCache, index: int,
                  par: Parallel | None, stash: list | None = None) -> torch.Tensor:
        from ..ops import kernels

        g = self.gpu
        par = par or self.par
        ids = kernels.ple_ids(step.token_ids, cache.ple_tokens, step.state_slots_t, step.query_start_t,
                              step.context_lens_t, self.n, self.heads_per_ngram, g["mult"], g["sizes"], g["offsets"],
                              self.eos)
        part = kernels.ple_gather(ids, g["h0"], g["h1"], g["packed"], g["scales"], g["row_base"], g["head_dim"],
                                  self._global_scale)
        if par.size > 1:
            emb = torch.zeros(part.shape[0], ids.shape[1] * g["head_dim"], dtype=part.dtype, device=part.device)
            emb[:, g["h0"] * g["head_dim"] : g["h1"] * g["head_dim"]] = part
            emb = par.all_reduce(emb)
        else:
            emb = part
        key = grouped_rms_norm(g["key"](emb).float(), self.norm_key, self.hidden, self.eps)
        value = g["value"](emb).float()
        query = grouped_rms_norm(h, self.norm_query, self.hidden, self.eps)
        gate = (key.view(-1, self.hc, self.hidden) * query.view(-1, self.hc, self.hidden)).sum(-1, keepdim=True)
        gate = gate / math.sqrt(self.hidden)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value[:, None, :]).reshape(h.shape[0], -1).contiguous()
        normed = grouped_rms_norm(gated, self.norm_conv, self.hidden, self.eps).contiguous()
        out = kernels.ple_conv(gated, normed, cache.ple_conv[index], self.conv, step.state_slots_t,
                               step.query_start_t, step.context_lens_t, self.taps, self.dilation)
        if stash is not None:
            stash.append((self, index, step.token_ids, normed))
            return out
        tok = cache.ple_tokens if self.last else cache.ple_tokens.clone()
        kernels.ple_state(tok, cache.ple_conv[index], step.token_ids, normed, step.state_slots_t,
                          step.query_start_t, step.context_lens_t, self.eos)
        return out

    def ngram_ids(self, tokens: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Table rows [T, heads] of one sequence's new tokens [T] after its previous n - 1 tokens [n - 1]."""
        eos = self.eos
        hist = torch.cat((previous.to(tokens.device), tokens))
        k = self.n - 1
        t = tokens.numel()
        shifted = [tokens.long()]
        for s in range(1, self.n):
            src = hist[k - s : k - s + t].long()
            bad = torch.zeros(t, dtype=torch.bool, device=tokens.device)
            for j in range(1, s + 1):
                bad |= hist[k - j : k - j + t] == eos
            shifted.append(torch.where(bad, torch.full_like(src, eos), src))
        blocks = []
        m = self.multipliers.to(tokens.device)
        for g in range(2, self.n + 1):
            a, b = (g - 2) * self.heads_per_ngram, (g - 1) * self.heads_per_ngram
            mixed = shifted[0] * m[0]
            for pos in range(1, g):
                mixed = torch.bitwise_xor(mixed, shifted[pos] * m[pos])
            sizes = self.vocab_sizes[a:b].to(tokens.device)
            blocks.append(torch.remainder(mixed[:, None], sizes[None, :]) + self.offsets[a:b].to(tokens.device))
        return torch.cat(blocks, dim=1)

    def __call__(self, h: torch.Tensor, step: StepInput, cache: ModelCache, index: int,
                 par: Parallel | None = None, stash: list | None = None) -> torch.Tensor:
        """What this layer adds to the streams h [T, hc x hidden] FP32; updates the sequences' PLE state, or with
        stash (GPU, speculative verify) appends what commit needs instead. par: the model's current layout (with the
        peer-to-peer all-reduce once the engine has set it up)."""
        if self.gpu is not None:
            return self._gpu_call(h, step, cache, index, par, stash)
        if stash is not None:
            raise NotImplementedError("the PLE layer's speculative verify runs on the GPU only")
        out = torch.empty_like(h)
        tokens = step.token_ids
        for b in range(len(step.context_lens)):
            q0, q1 = step.query_start[b], step.query_start[b + 1]
            if q1 == q0:
                continue
            slot, ctx = step.state_slots[b], step.context_lens[b]
            prev = cache.ple_tokens[slot] if ctx else torch.full((self.n - 1,), self.eos, dtype=torch.long)
            ids = self.ngram_ids(tokens[q0:q1], prev.to(tokens.device))
            emb = self.table.rows(ids).view(q1 - q0, -1).to(h.device)
            key = grouped_rms_norm(F.linear(emb, self.key_proj.dense), self.norm_key, self.hidden, self.eps)
            value = F.linear(emb, self.value_proj.dense)
            query = grouped_rms_norm(h[q0:q1], self.norm_query, self.hidden, self.eps)
            gate = (key.view(-1, self.hc, self.hidden) * query.view(-1, self.hc, self.hidden)).sum(-1, keepdim=True)
            gate = gate / math.sqrt(self.hidden)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gated = (torch.sigmoid(gate) * value[:, None, :]).reshape(q1 - q0, -1)
            normed = grouped_rms_norm(gated, self.norm_conv, self.hidden, self.eps)
            past = cache.ple_conv[index, slot].float() if ctx else torch.zeros(self.history, normed.shape[1],
                                                                               device=h.device)
            seq = torch.cat((past.to(h.device), normed))
            conv = torch.zeros_like(normed)
            for k in range(self.taps):
                lag = (self.taps - 1 - k) * self.dilation
                conv += self.conv[:, k] * seq[self.history - lag : self.history - lag + (q1 - q0)]
            out[q0:q1] = gated + F.silu(conv)
            cache.ple_conv[index, slot] = seq[-self.history :].to(cache.ple_conv.dtype)
            new_hist = torch.cat((prev.to(tokens.device), tokens[q0:q1].long()))[-(self.n - 1) :]
            cache.ple_tokens[slot] = new_hist.to(cache.ple_tokens.device)
        return out


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm in FP32 of the last dimension, times weight (already 1 + w)."""
    f = x.float()
    return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * weight


class Indexer:
    """The Qwen Sparse Attention indexer of one attention layer: indexer_heads query heads and one key head of
    indexer_head_dim, from one projection of the block input. Raw keys go into the cache (cache.idx_k on the CPU;
    on the GPU only the pooled block keys stay, see csrc/qsa.cu). A block of
    indexer_compress consecutive tokens has the key k_norm(mean of its raw keys), rotated at the block's first
    position; a query scores a block with sum over heads of relu(q . key) / sqrt(dim) and attends to the tokens of
    its top indexer_budget / indexer_compress complete blocks and of the unfinished last block (selected())."""

    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, device):
        e = cfg.exp
        self.heads, self.dim, self.eps = e.indexer_heads, e.indexer_head_dim, cfg.rms_norm_eps
        self.budget, self.compress = e.indexer_budget, e.indexer_compress
        self.proj = Linear(ckpt, [Source(f"{p}.index_qk_proj")], torch.float32, device, keep_dense=True)
        self.q_norm = (ckpt.tensor(f"{p}.q_layernorm.weight").float() + 1.0).to(device)
        self.k_norm = (ckpt.tensor(f"{p}.k_layernorm.weight").float() + 1.0).to(device)
        self.proj16 = None
        w = self.proj.dense
        if w.is_cuda and torch.equal(w.half().float(), w):
            from ..ops.kernels import DenseWeight

            self.proj16 = DenseWeight(w.half())
            self.proj.dense = w.half()
            torch.cuda.empty_cache()

    def prep(self, x: torch.Tensor, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
        """The GPU path of queries_and_keys for FP16 x: (queries FP16 [T, heads, dim], raw keys FP16 [T, dim]). With
        FP16 weights (exact for the checkpoint's weights): one product with FP32 sums (the dense GEMM kernel up to 64
        tokens, else cuBLAS tensor cores), then seventy_qsa_index_prep; else queries_and_keys."""
        if self.proj16 is None or x.dtype != torch.float16:
            iq, ik = self.queries_and_keys(x, cos, sin)
            return iq.to(torch.float16), ik.to(torch.float16)
        from ..ops import kernels

        if x.shape[0] <= 64:
            qk = kernels.dense_gemm(x, self.proj16)
        else:
            qk = torch.mm(x, self.proj.dense.T, out_dtype=torch.float32)
        return kernels.qsa_index_prep(qk, self.q_norm, cos, sin, self.heads, self.eps)

    def queries_and_keys(self, x: torch.Tensor, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
        """(normed, rotated queries [T, heads, dim], raw keys [T, dim]) of the block input x [T, hidden]."""
        qk = F.linear(x.float(), self.proj.dense.float())
        q = qk[:, : self.heads * self.dim].view(-1, self.heads, self.dim)
        k = qk[:, self.heads * self.dim :]
        q = _rms(q.reshape(-1, self.dim), self.q_norm, self.eps).view(-1, self.heads, self.dim)
        return ops.apply_rotary(q, cos, sin), k

    def selected(self, q: torch.Tensor, keys: torch.Tensor, cos_all, sin_all, first: int) -> torch.Tensor:
        """Token indices [n_q, budget + compress - 1] (-1: none) that the queries q [n_q, heads, dim] at positions
        first .. first + n_q - 1 attend to, from the raw keys [ctx, dim] of the sequence and the rotary tables of
        every position: all visible tokens while they fit in whole blocks of the budget, else the tokens of the best
        blocks and of the unfinished last block."""
        n_q, ctx = q.shape[0], keys.shape[0]
        c = self.compress
        top = self.budget // c
        width = self.budget + c - 1
        dev = keys.device
        visible = first + torch.arange(n_q, device=dev) + 1
        idx = torch.arange(width, device=dev)
        out = torch.where(idx[None, :] < visible[:, None], idx[None, :], -1)
        long = visible // c > top
        if not bool(long.any()):
            return out
        blocks = ctx // c
        pooled = keys[: blocks * c].float().view(blocks, c, self.dim).mean(1)
        pooled = _rms(pooled, self.k_norm, self.eps)
        starts = torch.arange(blocks, device=dev) * c
        pooled = ops.apply_rotary(pooled[:, None], cos_all[starts], sin_all[starts])[:, 0]
        rows = torch.nonzero(long).reshape(-1)
        scores = torch.relu(torch.einsum("qhd,bd->qbh", q[rows].float(), pooled.float())).sum(-1) / math.sqrt(self.dim)
        nb = visible[rows] // c
        scores = scores.masked_fill(torch.arange(blocks, device=dev)[None, :] >= nb[:, None], float("-inf"))
        best = torch.topk(scores, top, dim=1).indices
        tok = (best[:, :, None] * c + torch.arange(c, device=dev)).reshape(rows.numel(), -1)
        tail = nb[:, None] * c + torch.arange(c - 1, device=dev)[None, :]
        tail = torch.where(tail < visible[rows][:, None], tail, -1)
        out[rows] = torch.cat((tok, tail), dim=1).to(out.dtype)
        return out


class ExpAttention:
    """Gated full attention with the QSA indexer. While every sequence of a step fits in the budget (dense()), QSA is
    causal attention, and the GPU runs the attention kernels of AttentionLayer. Otherwise each query attends to the
    tokens Indexer.selected gives, in chunks of QUERY_CHUNK queries over the gathered keys and values (PyTorch, on
    the CPU and the GPU; the QSA kernels are not written yet). The indexer keys go into the cache in both cases."""

    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, par: Parallel, dtype, device, keep_dense):
        self.attn = AttentionLayer(cfg, ckpt, p, par, dtype, device, keep_dense)
        self.indexer = Indexer(cfg, ckpt, f"{p}.indexer", device)
        self.cfg = cfg

    def _kernels(self, x, step: StepInput, cache: ModelCache, layer: int, cos, sin) -> torch.Tensor:
        """The GPU path: attn_prep (queries, keys and values into the cache), the indexer queries and raw keys, then
        csrc/qsa.cu (pooled block keys from the raw keys and the slot's ring, selected positions in chunks of
        SELECT_ROWS rows (prefill: block scores from cuBLAS), sparse attention with the output gate)."""
        from ..ops import kernels

        a, e = self.attn, self.cfg.exp
        t = x.shape[0]
        qkv = a.qkv(x)
        ks, vs = cache.k_scale.get(layer), cache.v_scale.get(layer)
        q = kernels.attn_prep(qkv, a.q_norm, a.k_norm, step.positions, a.inv_freq, a.section, step.kv_slots_t,
                              cache.k[layer], cache.v[layer], a.heads, a.kv_heads, a.eps, ks, vs, rope_scale=a.rope_scale)
        iq16, raw = self.indexer.prep(x, cos, sin)
        kernels.qsa_pool(cache.pooled[layer], raw, cache.idx_ring[layer], step.state_slots_t, cache.positions,
                         step.block_tables, step.query_start_t, step.context_lens_t, t, self.indexer.k_norm,
                         a.inv_freq, a.section, self.cfg.rotary_dim, e.indexer_compress, cache.block_size,
                         self.indexer.eps, rope_scale=a.rope_scale)
        top = e.indexer_budget // e.indexer_compress
        ld = cache.num_blocks * cache.block_size // e.indexer_compress
        parts = []
        if t > PREFILL_ROWS and step.query_start and len(step.query_start) == len(step.context_lens) + 1:
            for b in range(len(step.context_lens)):
                q0, q1 = step.query_start[b], step.query_start[b + 1]
                if q1 == q0:
                    continue
                keys = self._pooled_keys(cache, layer, step, b, (step.context_lens[b] + q1 - q0) // e.indexer_compress)
                for r0 in range(q0, q1, SELECT_ROWS):
                    r1 = min(q1, r0 + SELECT_ROWS)
                    sel = kernels.qsa_select_prefill(iq16[r0:r1], keys, step.query_start_t, step.context_lens_t,
                                                     e.indexer_compress, top, r0)
                    parts.append(kernels.qsa_attention(q[r0:r1], qkv[r0:r1], cache.k[layer], cache.v[layer], ks, vs,
                                                       sel, step.block_tables, step.query_start_t, a.scale, row0=r0))
        else:
            for r0 in range(0, t, SELECT_ROWS):
                r1 = min(t, r0 + SELECT_ROWS)
                sel = kernels.qsa_select(iq16[r0:r1], cache.pooled[layer], step.block_tables, step.query_start_t,
                                         step.context_lens_t, e.indexer_compress, top, ld, cache.block_size, row0=r0)
                parts.append(kernels.qsa_attention(q[r0:r1], qkv[r0:r1], cache.k[layer], cache.v[layer], ks, vs, sel,
                                                   step.block_tables, step.query_start_t, a.scale, row0=r0))
        o = parts[0] if len(parts) == 1 else torch.cat(parts)
        return a.o_proj(a._owned(o))

    def _pooled_keys(self, cache: ModelCache, layer: int, step: StepInput, b: int, blocks: int) -> torch.Tensor:
        """The pooled block keys FP16 [blocks, dim] of blocks 0 .. blocks - 1 of sequence b, from the paged cache."""
        c, bs = self.cfg.exp.indexer_compress, cache.block_size
        pooled = cache.pooled[layer]
        pos = torch.arange(blocks, device=pooled.device) * c
        table = step.block_tables[b].to(pooled.device).long()
        idx = table[pos // bs] * (bs // c) + (pos % bs) // c
        return pooled.view(-1, pooled.shape[-1])[idx]

    def dense(self, step: StepInput) -> bool:
        """Whether every query of the step attends to all its visible tokens (each sequence has no more than
        indexer_budget tokens in whole blocks), so that QSA is plain causal attention."""
        e = self.cfg.exp
        top = e.indexer_budget // e.indexer_compress
        return all((step.context_lens[b] + step.query_start[b + 1] - step.query_start[b]) // e.indexer_compress <= top
                   for b in range(len(step.context_lens)))

    def __call__(self, x, step: StepInput, cache: ModelCache, layer: int, rope, slots, model) -> torch.Tensor:
        a = self.attn
        t, d = x.shape[0], a.head_dim
        cos, sin = rope
        if x.is_cuda and d == 256:
            return self._kernels(x, step, cache, layer, cos, sin)
        iq, ik = self.indexer.queries_and_keys(x, cos, sin)
        cache.idx_k[layer].view(-1, cache.idx_k[layer].shape[-1])[slots] = ik.to(cache.idx_k[layer].dtype)
        qkv = a.qkv(x)
        qg = qkv[:, : 2 * d * a.heads].view(t, a.heads, 2, d)
        q = qg[:, :, 0].contiguous()
        gate = qg[:, :, 1].reshape(t, a.heads * d)
        k = qkv[:, 2 * d * a.heads : 2 * d * a.heads + d * a.kv_heads].contiguous()
        v = qkv[:, 2 * d * a.heads + d * a.kv_heads :].reshape(t, a.kv_heads, d)
        q = ops.rms_norm(q.view(-1, d), a.q_norm, a.eps)[0].view(t, a.heads, d)
        k = ops.rms_norm(k.view(-1, d), a.k_norm, a.eps)[0].view(t, a.kv_heads, d)
        q = ops.apply_rotary(q, cos, sin)
        k = ops.apply_rotary(k, cos, sin)
        ks, vs = cache.k_scale.get(layer), cache.v_scale.get(layer)
        ops.write_kv(cache.k[layer], cache.v[layer], k, v, slots, ks, vs)
        out = torch.empty(t, a.heads, d, dtype=x.dtype, device=x.device)
        bs = step.block_size
        group = a.heads // a.kv_heads
        for b in range(len(step.context_lens)):
            q0, q1 = step.query_start[b], step.query_start[b + 1]
            if q1 == q0:
                continue
            before = step.context_lens[b]
            ctx = before + (q1 - q0)
            table = step.block_tables[b]
            pos = torch.arange(ctx, device=x.device)
            flat = table[pos // bs].long() * bs + pos % bs
            keys_i = cache.idx_k[layer].view(-1, cache.idx_k[layer].shape[-1])[flat]
            cos_all, sin_all = model.position_rope(cache, step, b, ctx)
            for c0 in range(q0, q1, QUERY_CHUNK):
                c1 = min(q1, c0 + QUERY_CHUNK)
                sel = self.indexer.selected(iq[c0:c1], keys_i, cos_all, sin_all, before + (c0 - q0))
                valid = sel >= 0
                rows = flat[sel.clamp(min=0).long()].reshape(-1)
                kk = _read_cache(cache.k[layer], ks, rows).float().view(c1 - c0, -1, a.kv_heads, d)
                vv = _read_cache(cache.v[layer], vs, rows).float().view(c1 - c0, -1, a.kv_heads, d)
                qg = q[c0:c1].float().view(c1 - c0, a.kv_heads, group, d)
                scores = torch.einsum("qgrd,qkgd->qgrk", qg, kk) * a.scale
                scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
                o = torch.einsum("qgrk,qkgd->qgrd", torch.softmax(scores, dim=-1), vv)
                out[c0:c1] = o.reshape(c1 - c0, a.heads, d).to(x.dtype)
        o = ops.sigmoid_gate(out.reshape(t, a.heads * d), gate)
        return a.o_proj(a._owned(o))


STREAMS = -2
QUERY_CHUNK = 64
SELECT_ROWS = 256
PREFILL_ROWS = 64


def _read_cache(c: torch.Tensor, scale: torch.Tensor | None, flat: torch.Tensor) -> torch.Tensor:
    rows = c.view(-1, *c.shape[2:])[flat]
    if scale is None:
        return rows
    return rows.float() * scale.view(-1, scale.shape[-1])[flat][..., None]


class ExpLayer:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, index: int, par: Parallel, dtype, device, keep_dense, eos,
                 prefix: str | None = None, kind: str | None = None):
        """Layer index of the model (prefix: the checkpoint prefix and kind of a layer outside the model's layers,
        such as the MTP layer)."""
        p = prefix or f"{ckpt.text_prefix}layers.{index}"
        self.kind = kind or cfg.layer_types[index]
        if self.kind in ("full_attention", "indexed_attention"):
            self.kind = "full_attention"
            self.mixer = ExpAttention(cfg, ckpt, f"{p}.self_attn", par, dtype, device, keep_dense)
        else:
            self.mixer = GatedDeltaNetLayer(cfg, ckpt, f"{p}.linear_attn", par, dtype, device, keep_dense)
        self.mlp = MoE(cfg, ckpt, f"{p}.mlp", par, dtype, device, keep_dense)
        self.attn_hc = HyperConnection(cfg, ckpt, f"{p}.attn_hyper_connection", dtype, device)
        self.mlp_hc = HyperConnection(cfg, ckpt, f"{p}.mlp_hyper_connection", dtype, device)
        self.ple = (PLE(cfg, ckpt, f"{p}.ple", device, dtype, eos, par, last=index == cfg.exp.ple_layers[-1])
                    if index in cfg.exp.ple_layers else None)


class Qwen4ExpTextModel(Qwen35TextModel):
    def __init__(self, cfg: ModelConfig, ckpt: Checkpoint, device: torch.device | str, dtype: torch.dtype = torch.float16,
                 par: Parallel = SINGLE, layers: int | None = None, keep_dense: bool = False):
        t = cfg.text
        self.cfg, self.par, self.dtype, self.device = t, par, dtype, torch.device(device)
        p = ckpt.text_prefix
        v0, v1 = par.span(t.vocab_size, VOCAB_ALIGN)
        self.vocab_span = (v0, v1)
        self.vocab_parts = [Parallel(r, par.size).span(t.vocab_size, VOCAB_ALIGN) for r in range(par.size)]
        self.embed = to_device(_to_dtype(ckpt.rows(f"{p}embed_tokens.weight", v0, v1), dtype), device)
        eos = t.eos_token_ids[0] if t.eos_token_ids else 0
        count = t.num_layers if layers is None else layers
        self.layers = [ExpLayer(t, ckpt, i, par, dtype, device, keep_dense, eos) for i in range(count)]
        half = t.rotary_dim // 2
        inv_freq, self.rope_scale = frequencies(t)
        self.inv_freq = inv_freq.to(device)
        idx = torch.arange(half)
        axis = torch.zeros(half, dtype=torch.long)
        axis[(idx % 3 == 1) & (idx < 3 * t.mrope_section[1])] = 1
        axis[(idx % 3 == 2) & (idx < 3 * t.mrope_section[2])] = 2
        self.rope_axis = axis.to(device)
        for layer in self.layers:
            if layer.kind == "full_attention":
                layer.mixer.attn.inv_freq = self.inv_freq
                layer.mixer.attn.rope_scale = self.rope_scale
        self.gdn = [layer.mixer for layer in self.layers if layer.kind != "full_attention"]
        for j, m in enumerate(self.gdn):
            m.gdn_index = j
        self.gdn_params = None
        if self.gdn and self.device.type == "cuda":
            stacked = []
            for name in ("conv_weight", "a_log", "dt_bias", "norm"):
                both = torch.stack([getattr(m, name) for m in self.gdn])
                for j, m in enumerate(self.gdn):
                    setattr(m, name, both[j])
                stacked.append(both)
            self.gdn_params = tuple(stacked)
        self.mixer = HyperConnection(t, ckpt, f"{p}hyper_connection_mixer", dtype, device, combine=False)
        self.lm_head_packed = None
        if ckpt.is_nvfp4("lm_head"):
            nv = Linear(ckpt, [Source("lm_head", ((v0, v1),))], dtype, device, keep_dense=self.device.type != "cuda")
            self.lm_head, self.lm_head_packed = nv.dense, nv.gemm
        else:
            self.lm_head = to_device(_to_dtype(ckpt.linear_part("lm_head", rows=(v0, v1)), dtype), device)
        if self.lm_head is not None and self.device.type == "cuda" and dtype == torch.float16:
            from ..ops.kernels import DenseWeight

            self.lm_head_packed = DenseWeight(self.lm_head)
            self.lm_head = None
            torch.cuda.empty_cache()

    def _inject(self, h: torch.Tensor, y: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        """h + the block output y added to every stream with its injection weight."""
        if h.is_cuda and y.dtype == torch.float16:
            from ..ops import kernels

            return kernels.hc_inject(h, y, inject, self.cfg.exp.hc_count)
        return h + (y.float()[:, None, :] * inject[:, :, None]).reshape(h.shape)

    def position_rope(self, cache: ModelCache, step: StepInput, b: int, ctx: int):
        """cos and sin of every position 0 .. ctx - 1 of sequence b, from the positions the cache keeps per token."""
        bs = step.block_size
        pos = torch.arange(ctx, device=cache.positions.device)
        flat = step.block_tables[b][pos // bs].long().to(cache.positions.device) * bs + pos % bs
        positions = cache.positions.view(-1, 3)[flat].T.long()
        cos, sin = ops.rotary_cos_sin(positions, self.inv_freq.cpu() if positions.device.type == "cpu" else self.inv_freq,
                                      self.cfg.mrope_section)
        return cos * self.rope_scale, sin * self.rope_scale

    @torch.inference_mode()
    def forward(self, step: StepInput, cache: ModelCache, capture: tuple[int, ...] = (), stash: dict | None = None,
                top: int = 0):
        """Run one step. Returns the logits of step.logit_rows (or head_top) and the captured streams: layer i in
        capture gives the streams after layer i, STREAMS the streams after the last layer (before the final mix,
        the MTP layer's input) and FINAL the final mix (the LM head's input), each for all rows.

        With stash (a dict, speculative verify): the Gated DeltaNet layers leave their state unchanged and keep their
        in_proj outputs, and the PLE layers leave their state unchanged and keep their inputs; commit() then writes
        both states for the accepted rows."""
        t = self.cfg
        e = t.exp
        x0 = self.embed_tokens(step.token_ids).float()
        for start, rows in step.embeds.items():
            x0[start : start + rows.shape[0]] = rows.float()
        h = x0.repeat(1, e.hc_count)
        if h.is_cuda:
            pos = step.positions.to(torch.float32)[self.rope_axis].T
            angles = pos * self.inv_freq
            rope = (torch.cos(angles) * self.rope_scale, torch.sin(angles) * self.rope_scale)
        else:
            cos, sin = ops.rotary_cos_sin(step.positions, self.inv_freq, t.mrope_section)
            rope = (cos * self.rope_scale, sin * self.rope_scale)
        if h.is_cuda:
            slots = step.kv_slots_t.long()
        else:
            slots = ops.kv_slots(step.query_start, step.context_lens, step.block_tables, step.block_size)
        cache.positions.view(-1, 3)[slots] = step.positions.T.to(cache.positions.dtype)
        captured = {}
        if stash is not None and self.gdn_params is not None and GDN_COMMIT_LAYERS:
            stash["y_all"] = torch.empty(len(self.gdn), h.shape[0], self.gdn[0].in_proj.out_features,
                                         dtype=torch.float16, device=h.device)
        ple_stash = [] if stash is not None else None
        for i, layer in enumerate(self.layers):
            if layer.ple is not None:
                h = h + layer.ple(h, step, cache, e.ple_layers.index(i), self.par, ple_stash)
            x, inject = layer.attn_hc(h)
            xin = x.to(self.dtype)
            if layer.kind == "full_attention":
                y = layer.mixer(xin, step, cache, i, rope, slots, self)
            elif stash is not None:
                y = layer.mixer(xin, step, cache, i, rope, slots, stash)
            else:
                y = layer.mixer(xin, step, cache, i, rope, slots)
            h = self._inject(h, self.par.all_reduce(y), inject)
            x, inject = layer.mlp_hc(h)
            h = self._inject(h, self.par.all_reduce(layer.mlp(x.to(self.dtype))), inject)
            if i in capture:
                captured[i] = h.clone()
        if ple_stash:
            stash["ple"] = ple_stash
        if STREAMS in capture:
            captured[STREAMS] = h
        x, _ = self.mixer(h)
        x = x.to(self.dtype)
        if FINAL in capture:
            captured[FINAL] = x
        if top:
            return self.head_top(x[step.logit_rows], top), captured
        return self.logits(x[step.logit_rows]), captured

    @torch.inference_mode()
    def commit(self, step: StepInput, cache: ModelCache, stash: dict, commit_lens: torch.Tensor) -> None:
        """After a speculative verify step: the Gated DeltaNet states (Qwen35TextModel.commit) and the PLE token and
        convolution states as if only the first commit_lens[b] new tokens of every sequence had run."""
        ple = stash.pop("ple", ())
        super().commit(step, cache, stash, commit_lens)
        from ..ops import kernels

        for layer, index, tokens, normed in ple:
            tok = cache.ple_tokens if layer.last else cache.ple_tokens.clone()
            kernels.ple_state(tok, cache.ple_conv[index], tokens, normed, step.state_slots_t, step.query_start_t,
                              step.context_lens_t, layer.eos, lens=commit_lens)
