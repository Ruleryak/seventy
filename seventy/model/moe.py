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

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(expert ids [tokens, k] int64, weights [tokens, k] FP32) of each token."""
        probs = torch.softmax(x.float() @ self.router.T, dim=-1)
        weights, ids = torch.topk(probs, self.top_k, dim=-1)
        if not self.cfg.norm_topk_prob:
            return ids, weights
        return ids, weights / weights.sum(dim=-1, keepdim=True)

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_down(ops.silu_mul(self.shared_gate_up(x)))
        return shared.float() * torch.sigmoid(x.float() @ self.shared_gate.T)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum of the MoE output (the all-reduce after the MLP completes it)."""
        t = x.shape[0]
        if not t:
            return x.new_zeros(0, x.shape[1])
        out = self._shared(x)
        ids, weights = self.route(x)
        if x.is_cuda and x.dtype == torch.float16 and t <= SMALL_TOKENS:
            from ..ops import kernels

            pairs = t * self.top_k
            plan = kernels.moe_plan(ids.reshape(-1).int(), self.e0, self.e1)
            h = kernels.moe_gemm(x, self.top_k, self._w(self.gate_up), self.gate_up.scales, self.gate_up.alpha, plan,
                                 pairs)
            a = ops.silu_mul(h)
            y = kernels.moe_gemm(a, 1, self._w(self.down), self.down.scales, self.down.alpha, plan, pairs,
                                 pair_weight=weights.reshape(-1).float().contiguous())
            return (out + y.view(t, self.top_k, -1).sum(dim=1)).to(x.dtype)
        return self._loop(x, ids, weights, out).to(x.dtype)

    @staticmethod
    def _w(stack: ExpertStack) -> torch.Tensor:
        return stack.packed if stack.nvfp4 else stack.weight

    def _loop(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """One dense product per picked local expert (prefill and the CPU)."""
        t = x.shape[0]
        flat = ids.reshape(-1)
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
