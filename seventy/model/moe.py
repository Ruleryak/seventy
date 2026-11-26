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
LARGE_ROWS = 8
EXPERT_OFFLOAD = 0.0


class ExpertStack:
    """The rank's experts of one projection, stacked: NVFP4 (packed, scales, alpha) or dense (weight)."""

    def __init__(self, packed=None, scales=None, alpha=None, weight=None):
        self.packed, self.scales, self.alpha, self.weight = packed, scales, alpha, weight

    gemm = None
    gemm_ptrs = None
    host = None

    @property
    def nvfp4(self) -> bool:
        return self.packed is not None or self.gemm is not None

    def to_gemm(self, on_gpu: int | None = None) -> None:
        """Repack the NVFP4 experts for seventy_moe_mma (each expert like a GemmWeight, stacked: wpk, wsc, alpha
        with the experts first) and drop the plain packed form. on_gpu: keep only that many experts (the first) in
        GPU memory and the others in pinned host memory, which the kernel reads over PCIe (expert offload); the
        kernel then finds every expert's weights through the pointer tables gemm_ptrs."""
        experts, n, half_k = self.packed.shape
        k = half_k * 2
        dev = self.packed.device
        on_gpu = experts if on_gpu is None else max(0, min(experts, on_gpu))
        first = nvfp4.pack_for_gemm([nvfp4.Nvfp4Weight(self.packed[0], self.scales[0], 1.0)])
        wpk = torch.empty((on_gpu, *first.wpk.shape), dtype=torch.uint8, device=dev)
        wsc = torch.empty((on_gpu, *first.wsc.shape), dtype=torch.uint8, device=dev)
        host = experts - on_gpu
        if host:
            self.host = (torch.empty((host, *first.wpk.shape), dtype=torch.uint8, pin_memory=True),
                         torch.empty((host, *first.wsc.shape), dtype=torch.uint8, pin_memory=True))
        alpha = torch.zeros(experts, first.alpha.numel(), dtype=torch.float32, device=dev)
        for e in range(experts):
            g = first if e == 0 else nvfp4.pack_for_gemm([nvfp4.Nvfp4Weight(self.packed[e], self.scales[e], 1.0)])
            if e < on_gpu:
                wpk[e], wsc[e] = g.wpk, g.wsc
            else:
                self.host[0][e - on_gpu].copy_(g.wpk)
                self.host[1][e - on_gpu].copy_(g.wsc)
            alpha[e, :n] = self.alpha[e] * nvfp4.GEMM_ALPHA_SHIFT
        self.gemm = nvfp4.GemmWeight(wpk, wsc, alpha.view(-1), n, k)
        self.gemm_experts = experts
        self.gemm_strides = (first.wpk.numel() // 16, first.wsc.numel() // 4)
        self.on_gpu = on_gpu
        if host:
            self.gemm_ptrs = (
                torch.tensor([self._where(e)[0].data_ptr() for e in range(experts)], dtype=torch.int64, device=dev),
                torch.tensor([self._where(e)[1].data_ptr() for e in range(experts)], dtype=torch.int64, device=dev))
        self.packed = self.scales = self.alpha = None

    def _where(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(wpk, wsc) of expert e after to_gemm: in GPU memory or, offloaded, in pinned host memory."""
        if e < self.on_gpu:
            return self.gemm.wpk[e], self.gemm.wsc[e]
        return self.host[0][e - self.on_gpu], self.host[1][e - self.on_gpu]

    @property
    def n(self) -> int:
        """Output features of each expert."""
        if self.gemm is not None:
            return self.gemm.n
        return (self.packed if self.packed is not None else self.weight).shape[1]

    def dense(self, e: int, dtype) -> torch.Tensor:
        """Expert e's weight [n, k] in dtype."""
        if not self.nvfp4:
            return self.weight[e].to(dtype)
        if self.gemm is not None:
            from ..ops import kernels

            g = self.gemm
            n_padded = g.alpha.numel() // self.gemm_experts
            wpk, wsc = (t.to(g.alpha.device) for t in self._where(e))
            one = nvfp4.GemmWeight(wpk, wsc, g.alpha[e * n_padded : (e + 1) * n_padded], g.n, g.k)
            return kernels.nvfp4_gemm_dequantize(one)[: g.n].to(dtype)
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
        elif self.device.type == "cuda" and dtype == torch.float16 and self.gate_up.packed is not None:
            n = self.e1 - self.e0
            offload = EXPERT_OFFLOAD if not p.startswith("mtp.") else 0.0
            on_gpu = n - int(round(n * offload)) if offload > 0 else None
            for stack in (self.gate_up, self.down):
                stack.to_gemm(on_gpu)
            torch.cuda.empty_cache()
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
        if self.router.dtype == torch.float16 and x.dtype == torch.float16 and x.is_cuda:
            logits = torch.mm(x, self.router.T, out_dtype=torch.float32)
        else:
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
        if self.shared_gate.dtype == torch.float16 and x.dtype == torch.float16 and x.is_cuda:
            g = torch.mm(x, self.shared_gate.T, out_dtype=torch.float32)
        else:
            g = x.float() @ self.shared_gate.float().T
        return shared.float() * torch.sigmoid(g * self.gate_scale)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum of the MoE output (the all-reduce after the MLP completes it)."""
        t = x.shape[0]
        if not t:
            return x.new_zeros(0, x.shape[1])
        if x.is_cuda and x.dtype == torch.float16 and (t <= SMALL_TOKENS or self.gate_up.gemm is not None):
            return self._small(x) if t <= SMALL_TOKENS else self._large(x)
        out = self._shared(x)
        ids, weights = self.route(x)
        return self._loop(x, ids, weights, out).to(x.dtype)

    def _logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(router logits FP32 [t, >= E] (strided), the shared expert's gate logit FP32 [t] (strided)) on the GPU:
        one dense GEMM of the FP16 gates up to SMALL_TOKENS tokens, else cuBLAS with FP32 sums; FP32 weights when
        FP16 would not hold them exactly."""
        if self.gates is not None and x.shape[0] <= SMALL_TOKENS:
            from ..ops import kernels

            lg = kernels.dense_gemm(x, self.gates, row_scale=self.gate_rows[: x.shape[0]])
            return lg, lg[:, self.cfg.num_experts]
        if self.router.dtype == torch.float16:
            logits = torch.mm(x, self.router.T, out_dtype=torch.float32) * self.gate_scale
            gate = torch.mm(x, self.shared_gate.T, out_dtype=torch.float32) * self.gate_scale
        else:
            logits = x.float() @ self.router.T
            gate = x.float() @ self.shared_gate.T
        return logits, gate[:, 0]

    def _small(self, x: torch.Tensor) -> torch.Tensor:
        """Up to SMALL_TOKENS tokens on the GPU, without host synchronization: the logits (_logits: with FP16 gates
        one product gives the router logits and the shared expert's gate logit; the BF16 weights times a power of
        two are exact in FP16, the tensor-core products are exact and the sums FP32, and the output scale 2^-e is
        exact), the route, the plan and the two grouped expert products, then moe_finish adds the gated shared
        expert."""
        from ..ops import kernels

        t, k, n = x.shape[0], self.top_k, self.cfg.num_experts
        logits, gate = self._logits(x)
        ids, weights = kernels.moe_route(logits, k, self.cfg.norm_topk_prob, experts=n)
        shared = self.shared_down(ops.silu_mul(self.shared_gate_up(x)))
        ids = ids.reshape(-1)
        pairs = t * k
        per = self._per(False)
        plan = kernels.moe_plan(ids, self.e0, self.e1, per)
        h = torch.empty(pairs, self.gate_up.n, dtype=torch.float16, device=x.device)
        self._experts(x, k, self.gate_up, plan, per, h)
        a = ops.silu_mul(h)
        y = torch.empty(pairs, x.shape[1], dtype=torch.float32, device=x.device)
        self._experts(a, 1, self.down, plan, per, y, pair_weight=weights.reshape(-1))
        return kernels.moe_finish(y, ids, k, self.e0, self.e1, shared, gate)

    def _large(self, x: torch.Tensor) -> torch.Tensor:
        """More than SMALL_TOKENS tokens on the GPU (prefill) with repacked NVFP4 experts: the same grouped expert
        products as _small with slots of 64 pairs, on rows in plan order (only this rank's pairs; one host
        synchronization for their number). FP16 experts (BF16 checkpoints) take _loop: cuBLAS per expert."""
        from ..ops import kernels

        t, k, n = x.shape[0], self.top_k, self.cfg.num_experts
        logits, gate = self._logits(x)
        ids, weights = kernels.moe_route(logits, k, self.cfg.norm_topk_prob, experts=n)
        shared = self.shared_down(ops.silu_mul(self.shared_gate_up(x)))
        ids = ids.reshape(-1)
        per = self._per(True)
        plan, rows, local = kernels.moe_plan_large(ids, self.e0, self.e1, per)
        m = max(1, int(local))
        h = torch.empty(m, self.gate_up.n, dtype=torch.float16, device=x.device)
        self._experts(x, k, self.gate_up, plan, per, h, compact=2)
        a = ops.silu_mul(h)
        y = torch.empty(m, x.shape[1], dtype=torch.float32, device=x.device)
        self._experts(a, 1, self.down, plan, per, y, pair_weight=weights.reshape(-1), compact=3)
        return kernels.moe_finish(y, ids, k, self.e0, self.e1, shared, gate, rows=rows)

    def _per(self, large: bool) -> int:
        """Pairs per slot of a plan: for the tensor-core kernel (experts after to_gemm) 8 (small batches, one 8-row
        tile) or 64 (prefill); for moe_gemm MOE_ROWS_PER_SLOT or LARGE_ROWS."""
        from ..ops import kernels

        if self.gate_up.gemm is not None:
            return 64 if large else 8
        return LARGE_ROWS if large else kernels.MOE_ROWS_PER_SLOT

    @staticmethod
    def _experts(x: torch.Tensor, row_div: int, stack: ExpertStack, plan: tuple, per: int, out: torch.Tensor,
                 pair_weight: torch.Tensor | None = None, compact: int = 0) -> torch.Tensor:
        """The grouped product of one projection of the plan's pairs into out: seventy_moe_mma for repacked NVFP4
        experts, else seventy_moe_gemm."""
        from ..ops import kernels

        if stack.gemm is not None:
            return kernels.moe_mma(x, row_div, stack, plan, out, per, warps=4, pair_weight=pair_weight,
                                   compact=compact)
        w = stack.packed if stack.packed is not None else stack.weight
        return kernels.moe_gemm(x, row_div, w, stack.scales, stack.alpha, plan, out.shape[0], pair_weight=pair_weight,
                                out=out, rows=per, compact=compact)

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
