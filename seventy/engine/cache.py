"""The model's per-sequence memory: a paged KV cache for the attention layers, and a slot of Gated DeltaNet
convolution and recurrent state per sequence for the linear-attention layers.

Each kind of memory is one stacked tensor over its layers, so copying a state slot or a KV block (for the prefix
cache) is one copy for all layers. The per-layer dictionaries are views into the stacks.
"""

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
        extra_attention_layers: tuple[int, ...] = (),
    ):
        """kv_dtype: the attention KV cache format, torch.int8 (with per-token, per-head FP32 scales) or the activation
        dtype. Default: int8 on CUDA, the activation dtype elsewhere.

        One extra block (index num_blocks) and one extra state slot (index num_slots) take the writes of padding
        rows in CUDA-graph steps.

        extra_attention_layers: layer indices past the model's layers with a KV cache of their own (the MTP
        drafter's attention layer), in the same blocks as the model's attention layers."""
        device = torch.device(device)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.scratch_block = num_blocks
        self.scratch_slot = num_slots
        self.kv_dtype = kv_dtype or (torch.int8 if device.type == "cuda" else dtype)
        kv_heads = par.heads(cfg.num_heads, cfg.num_kv_heads).num_kv
        gdn = par.heads(cfg.gdn_v_heads, cfg.gdn_k_heads)
        hk, hv = gdn.num_kv, gdn.num_heads
        conv_dim = 2 * hk * cfg.gdn_k_dim + hv * cfg.gdn_v_dim
        attn = tuple(cfg.attention_layers) + tuple(extra_attention_layers)
        na, nl = len(attn), len(cfg.linear_layers)
        kv_shape = (na, num_blocks + 1, block_size, kv_heads, cfg.head_dim)
        self.k_all = torch.zeros(kv_shape, dtype=self.kv_dtype, device=device)
        self.v_all = torch.zeros(kv_shape, dtype=self.kv_dtype, device=device)
        self.k_scale_all = self.v_scale_all = None
        if self.kv_dtype == torch.int8:
            self.k_scale_all = torch.zeros(kv_shape[:-1], dtype=torch.float32, device=device)
            self.v_scale_all = torch.zeros(kv_shape[:-1], dtype=torch.float32, device=device)
        self.conv_all = torch.zeros(nl, num_slots + 1, conv_dim, cfg.conv_kernel - 1, dtype=dtype, device=device)
        self.state_all = torch.zeros(nl, num_slots + 1, hv, cfg.gdn_k_dim, cfg.gdn_v_dim, dtype=torch.float32,
                                     device=device)
        self.k = {layer: self.k_all[i] for i, layer in enumerate(attn)}
        self.v = {layer: self.v_all[i] for i, layer in enumerate(attn)}
        self.k_scale: dict[int, torch.Tensor] = {}
        self.v_scale: dict[int, torch.Tensor] = {}
        if self.k_scale_all is not None:
            self.k_scale = {layer: self.k_scale_all[i] for i, layer in enumerate(attn)}
            self.v_scale = {layer: self.v_scale_all[i] for i, layer in enumerate(attn)}
        self.conv = {layer: self.conv_all[i] for i, layer in enumerate(cfg.linear_layers)}
        self.state = {layer: self.state_all[i] for i, layer in enumerate(cfg.linear_layers)}
        self.slot_peers: list = []

    def copy_slot(self, src: int, dst: int) -> None:
        """Copy the Gated DeltaNet state and convolution history of every layer from one slot to another."""
        self.state_all[:, dst].copy_(self.state_all[:, src])
        self.conv_all[:, dst].copy_(self.conv_all[:, src])
        for peer in self.slot_peers:
            peer.copy_slot(src, dst)

    def copy_block(self, src: int, dst: int) -> None:
        """Copy one KV block (all attention layers, keys, values and scales)."""
        self.k_all[:, dst].copy_(self.k_all[:, src])
        self.v_all[:, dst].copy_(self.v_all[:, src])
        if self.k_scale_all is not None:
            self.k_scale_all[:, dst].copy_(self.k_scale_all[:, src])
            self.v_scale_all[:, dst].copy_(self.v_scale_all[:, src])

    def apply(self, copies: list[tuple[str, int, int]]) -> None:
        for kind, src, dst in copies:
            if kind == "slot":
                self.copy_slot(src, dst)
            else:
                self.copy_block(src, dst)

    def nbytes(self) -> int:
        tensors = [self.k_all, self.v_all, self.conv_all, self.state_all]
        if self.k_scale_all is not None:
            tensors += [self.k_scale_all, self.v_scale_all]
        return sum(t.numel() * t.element_size() for t in tensors)
