"""The mixture-of-experts MLP of Qwen3.5-MoE models (for example Qwen3.6-35B-A3B).

A router picks experts_per_token of num_experts experts for each token: a softmax over all experts in FP32, the top
k, and the k weights divided by their sum (unless norm_topk_prob is false). Each expert is a SwiGLU MLP of width moe_intermediate_size. A shared
SwiGLU expert, scaled by sigmoid(shared_expert_gate(x)), adds to every token.

Tensor parallel: each rank keeps a contiguous part of the routed experts (expert parallel, split.balanced) and adds,
for every token, the weighted outputs of the picked experts that it keeps. The shared expert splits its columns over
the ranks like a dense MLP. Both are partial sums, which meet in the all-reduce after the MLP, as for a dense MLP.

Expert weights come per expert (experts.E.gate_proj, up_proj, down_proj: NVFP4 of compressed-tensors or ModelOpt,
or dense) or fused, as the BF16 checkpoints of Qwen store them (experts.gate_up_proj [E, 2 x width, hidden] and
experts.down_proj [E, hidden, width]). The rank stacks its experts: gate_up [E, 2 x width, hidden] (gate rows,
then up rows) and down [E, hidden, width], as NVFP4 (packed values, block scales, and alpha = 1 / global scale per
row) or as FP16 (FP32 on the CPU).

On the GPU, steps of up to SMALL_TOKENS tokens run on the grouped kernels of csrc/moe.cu (plan, gate_up, SwiGLU,
down), with no host synchronization, so they fit in CUDA graphs. Larger steps (prefill) and the CPU run one dense
product per picked expert.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .. import nvfp4, ops
from ..checkpoint import Checkpoint
from ..config import TextConfig
from ..parallel import MLP_ALIGN, Parallel
from ..split import balanced
from ..upload import to_device
from .layers import Linear, Source, _to_dtype

SMALL_TOKENS = 64


class ExpertStack:
    """The rank's experts of one projection, stacked: NVFP4 (packed, scales, alpha) or dense (weight)."""

    def __init__(self, packed=None, scales=None, alpha=None, weight=None):
        self.packed, self.scales, self.alpha, self.weight = packed, scales, alpha, weight

    @property
    def nvfp4(self) -> bool:
        return self.packed is not None

    def dense(self, e: int, dtype) -> torch.Tensor:
        """Expert e's weight [n, k] in dtype."""
        if not self.nvfp4:
            return self.weight[e].to(dtype)
        w = nvfp4.Nvfp4Weight(self.packed[e], self.scales[e], 1.0)
        if self.packed.is_cuda:
            values = ops.nvfp4_dequantize(w).float()
        else:
            values = nvfp4.dequantize(w, torch.float32)
        return (values * self.alpha[e][:, None]).to(dtype)


def _stack_nvfp4(experts: int, pieces, device) -> ExpertStack:
    """The NVFP4 stack of `experts` experts; pieces(e) gives the NVFP4 pieces (same k) of expert e's rows, in
    order. Each piece goes straight into one host tensor per part, which is copied to the device once."""
    packed = scales = alpha = None
    for e in range(experts):
        row = 0
        for p in pieces(e):
            if packed is None:
                n = sum(q.shape[0] for q in pieces(e))
                packed = torch.empty(experts, n, p.packed.shape[1], dtype=torch.uint8)
                scales = torch.empty(experts, n, p.scales.shape[1], dtype=torch.uint8)
                alpha = torch.empty(experts, n, dtype=torch.float32)
            r = p.shape[0]
            packed[e, row : row + r] = p.packed
            scales[e, row : row + r] = p.scales
            alpha[e, row : row + r] = 1.0 / p.global_scale
            row += r
    return ExpertStack(packed=to_device(packed, device), scales=to_device(scales, device),
                       alpha=to_device(alpha, device))


class MoE:
    def __init__(self, cfg: TextConfig, ckpt: Checkpoint, p: str, par: Parallel, dtype, device, keep_dense):
        self.cfg, self.par, self.dtype, self.device = cfg, par, dtype, torch.device(device)
        self.top_k = cfg.experts_per_token
        self.e0, self.e1 = balanced(cfg.num_experts, par.size, par.rank)
        self.router = to_device(ckpt.tensor(f"{p}.gate.weight").float(), device)
        self.shared_gate = to_device(ckpt.tensor(f"{p}.shared_expert_gate.weight").float(), device)
        s0, s1 = par.span(cfg.shared_expert_intermediate_size, MLP_ALIGN)
        sp = f"{p}.shared_expert"
        self.shared_gate_up = Linear(ckpt, [Source(f"{sp}.gate_proj", ((s0, s1),)), Source(f"{sp}.up_proj", ((s0, s1),))],
                                     dtype, device, keep_dense=keep_dense)
        self.shared_down = Linear(ckpt, [Source(f"{sp}.down_proj")], dtype, device, cols=(s0, s1),
                                  keep_dense=keep_dense)
        local = range(self.e0, self.e1)
        ep = f"{p}.experts"
        if f"{ep}.gate_up_proj" in ckpt:
            gate_up = _to_dtype(ckpt._slice(f"{ep}.gate_up_proj")[self.e0 : self.e1], dtype)
            down = _to_dtype(ckpt._slice(f"{ep}.down_proj")[self.e0 : self.e1], dtype)
            self.gate_up, self.down = ExpertStack(weight=to_device(gate_up, device)), ExpertStack(weight=to_device(down, device))
        elif ckpt.is_nvfp4(f"{ep}.{self.e0}.gate_proj"):
            n = self.e1 - self.e0
            gate_up = _stack_nvfp4(n, lambda e: [ckpt.nvfp4_part(f"{ep}.{self.e0 + e}.{m}_proj") for m in ("gate", "up")],
                                   device)
            down = _stack_nvfp4(n, lambda e: [ckpt.nvfp4_part(f"{ep}.{self.e0 + e}.down_proj")], device)
            self.gate_up, self.down = gate_up, down
        else:
            gate_up = torch.stack([_to_dtype(torch.cat((ckpt.linear_part(f"{ep}.{e}.gate_proj"),
                                                        ckpt.linear_part(f"{ep}.{e}.up_proj"))), dtype) for e in local])
            down = torch.stack([_to_dtype(ckpt.linear_part(f"{ep}.{e}.down_proj"), dtype) for e in local])
            self.gate_up, self.down = ExpertStack(weight=to_device(gate_up, device)), ExpertStack(weight=to_device(down, device))
        if keep_dense and self.gate_up.nvfp4:
            n = self.e1 - self.e0
            self.gate_up = ExpertStack(weight=torch.stack([self.gate_up.dense(e, dtype) for e in range(n)]))
            self.down = ExpertStack(weight=torch.stack([self.down.dense(e, dtype) for e in range(n)]))
        self.width = cfg.moe_intermediate_size
        self.gates = None
        self.gate_scale = 1.0
        if self.device.type == "cuda" and dtype == torch.float16:
            self.use_fp16_gates()

    def use_fp16_gates(self) -> bool:
        """Keep the router and the shared expert's gate (FP32 [E, H] and [1, H]) as FP16 times 2^e when that is
        exact (BF16 weights whose range fits), for the one-product route of _small. Returns whether it is."""
        from ..ops.kernels import DenseWeight

        both = torch.cat((self.router.float(), self.shared_gate.float()))
        top = both.abs().max().item()
        e = math.floor(math.log2(32768.0 / top)) if top > 0 else 0
        scaled = both * 2.0**e
        if not torch.equal(scaled.half().float(), scaled):
            return False
        self.gates = DenseWeight(scaled.half())
        self.gate_scale = 2.0**-e
        self.gate_rows = torch.full((SMALL_TOKENS,), self.gate_scale, device=both.device)
        n = self.router.shape[0]
        self.router, self.shared_gate = scaled[:n].half(), scaled[n:].half()
        torch.cuda.empty_cache()
        return True

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(expert ids [tokens, k] int64, or int32 from the kernel, weights [tokens, k] FP32) of each token. The router
        weights are FP32, or FP16 times 2^e (gate_scale = 2^-e) when that holds the BF16 weights exactly (with
        self.gates); the logits are FP32 either way."""
        logits = x.float() @ self.router.float().T
        if self.gate_scale != 1.0:
            logits = logits * self.gate_scale
        if logits.is_cuda:
            from ..ops import kernels

            return kernels.moe_route(logits, self.top_k, self.cfg.norm_topk_prob)
        probs = torch.softmax(logits, dim=-1)
        weights, ids = torch.topk(probs, self.top_k, dim=-1)
        if not self.cfg.norm_topk_prob:
            return ids, weights
        return ids, weights / weights.sum(dim=-1, keepdim=True)

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_down(ops.silu_mul(self.shared_gate_up(x)))
        return shared.float() * torch.sigmoid((x.float() @ self.shared_gate.float().T) * self.gate_scale)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum of the MoE output (the all-reduce after the MLP completes it)."""
        t = x.shape[0]
        if not t:
            return x.new_zeros(0, x.shape[1])
        if x.is_cuda and x.dtype == torch.float16 and t <= SMALL_TOKENS and self.gates is not None:
            return self._small(x)
        out = self._shared(x)
        ids, weights = self.route(x)
        if x.is_cuda and x.dtype == torch.float16 and t <= SMALL_TOKENS:
            from ..ops import kernels

            pairs = t * self.top_k
            plan = kernels.moe_plan(ids.reshape(-1), self.e0, self.e1)
            h = kernels.moe_gemm(x, self.top_k, self._w(self.gate_up), self.gate_up.scales, self.gate_up.alpha, plan,
                                 pairs)
            a = ops.silu_mul(h)
            y = kernels.moe_gemm(a, 1, self._w(self.down), self.down.scales, self.down.alpha, plan, pairs,
                                 pair_weight=weights.reshape(-1).float().contiguous())
            return (out + y.view(t, self.top_k, -1).sum(dim=1)).to(x.dtype)
        return self._loop(x, ids, weights, out).to(x.dtype)

    def _small(self, x: torch.Tensor) -> torch.Tensor:
        """Up to SMALL_TOKENS tokens on the GPU, without host synchronization: one FP16 product gives the router
        logits and the shared expert's gate logit (the BF16 weights times a power of two are exact in FP16, the
        tensor-core products are exact and the sums FP32, and the output scale 2^-e is exact), then the route,
        the plan and the two grouped expert products, and moe_finish adds the gated shared expert."""
        from ..ops import kernels

        t, k, n = x.shape[0], self.top_k, self.cfg.num_experts
        logits = kernels.dense_gemm(x, self.gates, row_scale=self.gate_rows[:t])
        ids, weights = kernels.moe_route(logits, k, self.cfg.norm_topk_prob, experts=n)
        shared = self.shared_down(ops.silu_mul(self.shared_gate_up(x)))
        ids = ids.reshape(-1)
        pairs = t * k
        plan = kernels.moe_plan(ids, self.e0, self.e1)
        h = torch.empty(pairs, self._w(self.gate_up).shape[1], dtype=torch.float16, device=x.device)
        kernels.moe_gemm(x, k, self._w(self.gate_up), self.gate_up.scales, self.gate_up.alpha, plan, pairs, out=h)
        a = ops.silu_mul(h)
        y = torch.empty(pairs, x.shape[1], dtype=torch.float32, device=x.device)
        kernels.moe_gemm(a, 1, self._w(self.down), self.down.scales, self.down.alpha, plan, pairs,
                         pair_weight=weights.reshape(-1), out=y)
        return kernels.moe_finish(y, ids, k, self.e0, self.e1, shared, logits[:, n])

    @staticmethod
    def _w(stack: ExpertStack) -> torch.Tensor:
        return stack.packed if stack.nvfp4 else stack.weight

    def _loop(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """One dense product per picked local expert (prefill and the CPU)."""
        t = x.shape[0]
        flat = ids.reshape(-1).long()
        token = torch.arange(t, device=x.device).repeat_interleave(self.top_k)
        mine = (flat >= self.e0) & (flat < self.e1)
        local = flat[mine] - self.e0
        token, weight = token[mine], weights.reshape(-1)[mine]
        order = torch.argsort(local, stable=True)
        local, token, weight = local[order], token[order], weight[order]
        counts = torch.bincount(local, minlength=self.e1 - self.e0).tolist()
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            rows = token[start : start + n]
            h = F.linear(x[rows], self.gate_up.dense(e, x.dtype))
            y = F.linear(ops.silu_mul(h), self.down.dense(e, x.dtype))
            out.index_add_(0, rows, y.float() * weight[start : start + n, None])
            start += n
        return out
