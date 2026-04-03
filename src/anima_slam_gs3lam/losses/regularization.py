"""Regularization blocks for GS3LAM."""

from __future__ import annotations

import torch


def depth_adaptive_scale_regularization(
    scales: torch.Tensor,
    *,
    num_std: float = 2.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Regularize overly large and overly small Gaussian scales.

    The public GS3LAM repo applies a distribution-based penalty over Gaussian scale values.
    This function keeps that behavior explicit and standalone for reuse in mapping losses.
    """

    if scales.ndim != 2:
        raise ValueError("scales must have shape [N, D]")
    if scales.shape[0] == 0:
        zero = torch.tensor(0.0, dtype=scales.dtype, device=scales.device)
        return {"big_gaussian_reg": zero, "small_gaussian_reg": zero, "loss": zero}

    positive_scales = scales.clamp_min(eps)
    max_scales = positive_scales.max(dim=-1).values
    min_scales = positive_scales.min(dim=-1).values

    log_max_scales = torch.log(max_scales)
    log_min_scales = torch.log(min_scales)
    upper_limit = log_max_scales.mean() + num_std * log_max_scales.std(unbiased=False)
    lower_limit = log_min_scales.mean() - num_std * log_min_scales.std(unbiased=False)

    big_mask = log_max_scales > upper_limit
    small_mask = log_min_scales < lower_limit

    zero = torch.tensor(0.0, dtype=scales.dtype, device=scales.device)
    big_reg = max_scales[big_mask].mean() if torch.any(big_mask) else zero
    small_reg = (-torch.log(min_scales[small_mask])).mean() if torch.any(small_mask) else zero
    return {
        "big_gaussian_reg": big_reg,
        "small_gaussian_reg": small_reg,
        "loss": big_reg + small_reg,
    }
