"""Rotary frequencies of a text config: the default rotary, or YaRN (context extension past the trained length).

YaRN as in transformers (modeling_rope_utils._compute_yarn_parameters): frequencies between extrapolation (the
original ones) and interpolation (divided by the factor), with a linear ramp between the correction dimensions of
beta_fast and beta_slow rotations over the original length; cos and sin are multiplied by the attention factor
(0.1 ln(factor) + 1 unless the config gives it). Computed in FP64, returned in FP32.
"""

from __future__ import annotations

import math

import torch

from .config import TextConfig


def frequencies(t: TextConfig) -> tuple[torch.Tensor, float]:
    """(inv_freq FP32 [rotary_dim / 2] on the CPU, the factor for cos and sin) of t."""
    dim = t.rotary_dim
    base = t.rope_theta
    pos = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    if t.rope_type != "yarn":
        return (1.0 / pos).float(), 1.0
    factor = t.rope_factor
    original = t.rope_original_max

    def correction(rotations: float) -> float:
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction(t.rope_beta_fast), correction(t.rope_beta_slow)
    if t.rope_truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low), 0, 1)
    extrapolation = 1 - ramp
    inv_freq = (1.0 / (factor * pos)) * (1 - extrapolation) + (1.0 / pos) * extrapolation
    return inv_freq.float(), float(t.rope_attention_factor)

