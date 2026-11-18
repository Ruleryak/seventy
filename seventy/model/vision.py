"""The Qwen3.5 vision tower: an image to LLM input embeddings, one row per 2 x 2 block of 16 x 16 pixel patches.

For an image of h x w patches (after the resize in seventy.multimodal):
  1. pixels are normalized to [-1, 1] and cut into patches in block-major order (the 4 patches of each 2 x 2 block
     are consecutive rows; blocks in raster order); each patch is repeated over the 2 temporal frames and goes
     through the linear patch embedding (the checkpoint's Conv3d);
  2. the learned 48 x 48 position table, resampled bilinearly (corners aligned) to h x w, is added;
  3. 27 pre-norm transformer blocks: LayerNorm, attention over all patches of the image (not causal) with a 2D
     rotary embedding (height angles then width angles, neox halves over the full head), LayerNorm, MLP (GELU, tanh
     approximation);
  4. merger: LayerNorm per patch, the 4 patches of a block concatenated, Linear, GELU (exact), Linear to the LLM
     width.

Numerics: FP16 weights and GEMM inputs. The residual stream is FP32: it reaches about 7600 in an outlier channel,
where FP16 has a spacing of 4. The GEMMs that add to it give FP32 output. Attention uses PyTorch's memory-efficient
SDPA kernel, which runs on sm70 without materializing the score matrix.

Tensor parallel: heads and MLP columns are split over the ranks (qkv, fc1 and the merger's first Linear by output
rows; proj, fc2 and the merger's second Linear by input columns). Partial sums are all-reduced in FP32, then the
bias is added once, so every rank ends with the same embeddings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..checkpoint import Checkpoint
from ..config import VisionConfig
from ..parallel import SINGLE, Parallel

_EPS = 1e-6


def _f16(t: torch.Tensor, device) -> torch.Tensor:
    return t.to(torch.float16).to(device).contiguous()


def _f32(t: torch.Tensor, device) -> torch.Tensor:
    return t.float().to(device).contiguous()


def block_major(t: torch.Tensor, h: int, w: int, merge: int) -> torch.Tensor:
    """Rows of an [h, w, ...] raster grid in block-major order (merge x merge blocks, raster order inside)."""
    rest = t.shape[2:]
    return t.reshape(h // merge, merge, w // merge, merge, *rest).transpose(1, 2).reshape(h * w, *rest)


def patchify(image: torch.Tensor, patch: int, merge: int, temporal: int) -> torch.Tensor:
    """uint8 [H, W, 3] (an image: its frame repeated over the temporal patch) or [temporal, H, W, 3] (video
    frames), sides multiples of patch x merge, to FP32 patches [h * w, 3 * temporal * patch * patch]: block-major
    rows, each patch's (channel, frame, row, col) values flattened."""
    frames = image[None] if image.dim() == 3 else image
    hh, ww = frames.shape[1] // patch, frames.shape[2] // patch
    x = (frames.permute(0, 3, 1, 2).float() - 127.5) / 127.5
    x = x.reshape(frames.shape[0], 3, hh // merge, merge, patch, ww // merge, merge, patch)
    x = x.permute(2, 5, 3, 6, 1, 0, 4, 7)
    if frames.shape[0] == 1:
        x = x.expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    elif frames.shape[0] != temporal:
        raise ValueError(f"a video group needs {temporal} frames, not {frames.shape[0]}")
    return x.reshape(hh * ww, 3 * temporal * patch * patch)


class VisionBlock:
    def __init__(self, ckpt: Checkpoint, p: str, cfg: VisionConfig, par: Parallel, device):
        d = cfg.hidden_size
        hd = d // cfg.num_heads
        h0, h1 = par.span(cfg.num_heads)
        i0, i1 = par.span(cfg.intermediate_size)
        rows = torch.cat([torch.arange(s * d + h0 * hd, s * d + h1 * hd) for s in range(3)])
        self.heads, self.head_dim = h1 - h0, hd
        self.norm1 = (_f32(ckpt.tensor(f"{p}.norm1.weight"), device), _f32(ckpt.tensor(f"{p}.norm1.bias"), device))
        self.norm2 = (_f32(ckpt.tensor(f"{p}.norm2.weight"), device), _f32(ckpt.tensor(f"{p}.norm2.bias"), device))
        self.qkv_w = _f16(ckpt.tensor(f"{p}.attn.qkv.weight")[rows], device)
        self.qkv_b = _f16(ckpt.tensor(f"{p}.attn.qkv.bias")[rows], device)
        self.proj_w = _f16(ckpt.tensor(f"{p}.attn.proj.weight")[:, h0 * hd : h1 * hd], device)
        self.proj_b = _f32(ckpt.tensor(f"{p}.attn.proj.bias"), device)
        self.fc1_w = _f16(ckpt.rows(f"{p}.mlp.linear_fc1.weight", i0, i1), device)
        self.fc1_b = _f16(ckpt.tensor(f"{p}.mlp.linear_fc1.bias")[i0:i1], device)
        self.fc2_w = _f16(ckpt.cols(f"{p}.mlp.linear_fc2.weight", i0, i1), device)
        self.fc2_b = _f32(ckpt.tensor(f"{p}.mlp.linear_fc2.bias"), device)
        self.dim = d


class VisionTower:
    def __init__(self, cfg: VisionConfig, ckpt: Checkpoint, device: torch.device | str, par: Parallel = SINGLE,
                 prefix: str = "model.visual."):
        if cfg.hidden_size % cfg.num_heads or (cfg.hidden_size // cfg.num_heads) % 4:
            raise ValueError("vision head size must be a multiple of 4")
        self.cfg, self.par, self.device = cfg, par, torch.device(device)
        dev = self.device
        p = prefix
        self.blocks = [VisionBlock(ckpt, f"{p}blocks.{i}", cfg, par, dev) for i in range(cfg.depth)]
        w = ckpt.tensor(f"{p}patch_embed.proj.weight")
        self.patch_w = _f16(w.reshape(w.shape[0], -1), dev)
        self.patch_b = _f32(ckpt.tensor(f"{p}patch_embed.proj.bias"), dev)
        self.side = int(round(cfg.num_position_embeddings ** 0.5))
        if self.side * self.side != cfg.num_position_embeddings:
            raise ValueError("the vision position table is not square")
        self.pos_table = _f32(ckpt.tensor(f"{p}pos_embed.weight"), dev).view(self.side, self.side, -1)
        m = cfg.spatial_merge_size
        wide = cfg.hidden_size * m * m
        j0, j1 = par.span(wide)
        self.merge_norm = (_f32(ckpt.tensor(f"{p}merger.norm.weight"), dev),
                           _f32(ckpt.tensor(f"{p}merger.norm.bias"), dev))
        self.merge_fc1_w = _f16(ckpt.rows(f"{p}merger.linear_fc1.weight", j0, j1), dev)
        self.merge_fc1_b = _f16(ckpt.tensor(f"{p}merger.linear_fc1.bias")[j0:j1], dev)
        self.merge_fc2_w = _f16(ckpt.cols(f"{p}merger.linear_fc2.weight", j0, j1), dev)
        self.merge_fc2_b = _f32(ckpt.tensor(f"{p}merger.linear_fc2.bias"), dev)
        axis_dim = cfg.hidden_size // cfg.num_heads // 2
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=dev) / axis_dim))

    def tokens(self, grid: tuple[int, int, int]) -> int:
        t, h, w = grid
        return t * h * w // (self.cfg.spatial_merge_size ** 2)

    def _positions(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resampled position table [h * w, dim] (FP32) and rotary cos, sin [h * w, head_dim], block-major."""
        dev, side, m = self.device, self.side, self.cfg.spatial_merge_size

        def taps(size: int):
            idx = torch.arange(size, dtype=torch.float32, device=dev)
            src = idx * (side - 1) / max(size - 1, 1)
            lo = torch.floor(src)
            frac = src - lo
            lo = lo.long()
            return lo.clamp(0, side - 1), (lo + 1).clamp(0, side - 1), 1.0 - frac, frac

        r0, r1, rw0, rw1 = taps(h)
        c0, c1, cw0, cw1 = taps(w)
        t = self.pos_table
        rows = rw0[:, None, None] * t[r0] + rw1[:, None, None] * t[r1]
        grid = cw0[None, :, None] * rows[:, c0] + cw1[None, :, None] * rows[:, c1]
        pos = block_major(grid, h, w, m)
        hp = torch.arange(h, device=dev)[:, None].expand(h, w)
        wp = torch.arange(w, device=dev)[None, :].expand(h, w)
        hw = block_major(torch.stack([hp, wp], dim=-1), h, w, m).float()
        ang = torch.cat([hw[:, :1] * self.inv_freq, hw[:, 1:] * self.inv_freq], dim=-1)
        ang = torch.cat([ang, ang], dim=-1)
        return pos, ang.cos(), ang.sin()

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.par.size > 1:
            torch.distributed.all_reduce(x)
        return x

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        half = xf.shape[-1] // 2
        rot = torch.cat([-xf[..., half:], xf[..., :half]], dim=-1)
        return (xf * cos[:, None] + rot * sin[:, None]).half()

    def _block(self, blk: VisionBlock, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = F.layer_norm(x, (blk.dim,), blk.norm1[0], blk.norm1[1], _EPS).half()
        qkv = F.linear(h, blk.qkv_w, blk.qkv_b).view(n, 3, blk.heads, blk.head_dim)
        q = self._rotate(qkv[:, 0], cos, sin).transpose(0, 1)[None]
        k = self._rotate(qkv[:, 1], cos, sin).transpose(0, 1)[None]
        v = qkv[:, 2].transpose(0, 1)[None]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            o = F.scaled_dot_product_attention(q, k, v)
        o = o[0].transpose(0, 1).reshape(n, blk.heads * blk.head_dim)
        x = x + self._reduce(torch.mm(o, blk.proj_w.T, out_dtype=torch.float32)) + blk.proj_b
        h = F.layer_norm(x, (blk.dim,), blk.norm2[0], blk.norm2[1], _EPS).half()
        h = F.gelu(F.linear(h, blk.fc1_w, blk.fc1_b), approximate="tanh")
        return x + self._reduce(torch.mm(h, blk.fc2_w.T, out_dtype=torch.float32)) + blk.fc2_b

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """uint8 [H, W, 3] image or [2, H, W, 3] video frame pair (already resized) to FP16 embeddings
        [tokens, out_hidden_size] in LLM order."""
        c = self.cfg
        p, m = c.patch_size, c.spatial_merge_size
        if image.dtype != torch.uint8 or image.dim() not in (3, 4) or image.shape[-1] != 3:
            raise ValueError("the image must be uint8 [height, width, 3] or [frames, height, width, 3]")
        if image.shape[-3] % (p * m) or image.shape[-2] % (p * m):
            raise ValueError(f"image sides must be multiples of {p * m}")
        image = image.to(self.device, non_blocking=True)
        h, w = image.shape[-3] // p, image.shape[-2] // p
        patches = patchify(image, p, m, c.temporal_patch_size).half()
        x = torch.mm(patches, self.patch_w.T, out_dtype=torch.float32) + self.patch_b
        pos, cos, sin = self._positions(h, w)
        x = x + pos
        for blk in self.blocks:
            x = self._block(blk, x, cos, sin)
        y = F.layer_norm(x, (c.hidden_size,), self.merge_norm[0], self.merge_norm[1], _EPS).half()
        y = y.view(-1, c.hidden_size * m * m)
        y = F.gelu(F.linear(y, self.merge_fc1_w, self.merge_fc1_b))
        y = self._reduce(torch.mm(y, self.merge_fc2_w.T, out_dtype=torch.float32)) + self.merge_fc2_b
        return y.half()
