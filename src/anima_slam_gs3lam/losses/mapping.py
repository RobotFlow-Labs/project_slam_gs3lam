"""Mapping loss with semantic and DSR terms."""

from __future__ import annotations

import torch

from anima_slam_gs3lam.losses.regularization import depth_adaptive_scale_regularization
from anima_slam_gs3lam.rendering.rasterizer import RenderOutputs


def mapping_loss(
    render: RenderOutputs,
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor,
    semantic_logits: torch.Tensor,
    target_semantic: torch.Tensor,
    scales: torch.Tensor,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weights is None:
        weights = {
            "color": 0.5,
            "depth": 1.0,
            "semantic": 0.01,
            "big_scale": 0.01,
            "small_scale": 0.001,
        }

    valid_mask = target_depth > 0
    color_term = torch.abs(render.rgb - target_rgb).mean()
    if torch.any(valid_mask):
        depth_term = torch.abs(render.depth - target_depth)[valid_mask].mean()
    else:
        depth_term = target_depth.sum() * 0.0

    semantic_loss = torch.nn.functional.cross_entropy(
        semantic_logits.unsqueeze(0),
        target_semantic.unsqueeze(0).long(),
    )
    semantic_term = semantic_loss / torch.log(torch.tensor(semantic_logits.shape[0], device=semantic_logits.device, dtype=semantic_logits.dtype))

    reg_terms = depth_adaptive_scale_regularization(scales)
    total = (
        weights["color"] * color_term
        + weights["depth"] * depth_term
        + weights["semantic"] * semantic_term
        + weights["big_scale"] * reg_terms["big_gaussian_reg"]
        + weights["small_scale"] * reg_terms["small_gaussian_reg"]
    )
    return total, {
        "color": color_term,
        "depth": depth_term,
        "semantic": semantic_term,
        "big_scale": reg_terms["big_gaussian_reg"],
        "small_scale": reg_terms["small_gaussian_reg"],
    }
