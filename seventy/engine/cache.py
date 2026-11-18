"""The model's per-sequence memory: a paged KV cache for the attention layers, and a slot of Gated DeltaNet
convolution and recurrent state per sequence for the linear-attention layers."""

from __future__ import annotations

import torch

from ..config import TextConfig
from ..parallel import SINGLE, Parallel


class ModelCache:
    def __init__(
        self,
        cfg: TextConfig,
        num_blocks: int,
        block_size: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device | str,
        par: Parallel = SINGLE,
        kv_dtype: torch.dtype | None = None,
    ):
        """kv_dtype: the attention KV cache format, torch.int8 (with per-token, per-head FP32 scales) or the activation
        dtype. Default: int8 on CUDA, the activation dtype elsewhere."""
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        kv_heads = cfg.num_kv_heads // par.size
        hk = cfg.gdn_k_heads // par.size
        hv = cfg.gdn_v_heads // par.size
        conv_dim = 2 * hk * cfg.gdn_k_dim + hv * cfg.gdn_v_dim
        device = torch.device(device)
        self.kv_dtype = kv_dtype or (torch.int8 if device.type == "cuda" else dtype)
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.k_scale: dict[int, torch.Tensor] = {}
        self.v_scale: dict[int, torch.Tensor] = {}
        self.conv: dict[int, torch.Tensor] = {}
        self.state: dict[int, torch.Tensor] = {}
        for layer in cfg.attention_layers:
            shape = (num_blocks, block_size, kv_heads, cfg.head_dim)
            self.k[layer] = torch.zeros(shape, dtype=self.kv_dtype, device=device)
            self.v[layer] = torch.zeros(shape, dtype=self.kv_dtype, device=device)
            if self.kv_dtype == torch.int8:
                self.k_scale[layer] = torch.zeros(shape[:-1], dtype=torch.float32, device=device)
                self.v_scale[layer] = torch.zeros(shape[:-1], dtype=torch.float32, device=device)
        for layer in cfg.linear_layers:
            self.conv[layer] = torch.zeros(num_slots, conv_dim, cfg.conv_kernel - 1, dtype=dtype, device=device)
            self.state[layer] = torch.zeros(num_slots, hv, cfg.gdn_k_dim, cfg.gdn_v_dim, dtype=torch.float32,
                                            device=device)

    def nbytes(self) -> int:
        tensors = [*self.k.values(), *self.v.values(), *self.k_scale.values(), *self.v_scale.values(),
                   *self.conv.values(), *self.state.values()]
        return sum(t.numel() * t.element_size() for t in tensors)
