"""Tracking loss from observable regions."""

from __future__ import annotations

import torch

from anima_slam_gs3lam.rendering.rasterizer import RenderOutputs
from anima_slam_gs3lam.tracking.tracker import compute_observed_mask


def tracking_loss(
    render: RenderOutputs,
    target_rgb: torch.Tensor,
    target_depth: torch.Tensor,
    semantic_logits: torch.Tensor,
    target_semantic: torch.Tensor,
    *,
    weights: dict[str, float] | None = None,
    tau_obs: float = 0.99,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weights is None:
        weights = {"color": 0.5, "depth": 1.0, "semantic": 0.001}

    observed_mask = compute_observed_mask(render.opacity[0], render.depth[0], target_depth[0], tau_obs=tau_obs)
    if not torch.any(observed_mask):
        zero = target_rgb.sum() * 0.0
        return zero, {"color": zero, "depth": zero, "semantic": zero}

    color_mask = observed_mask.unsqueeze(0).expand_as(target_rgb)
    color_term = torch.abs(render.rgb - target_rgb)[color_mask].mean()
    depth_term = torch.abs(render.depth - target_depth)[observed_mask.unsqueeze(0)].mean()

    semantic_loss = torch.nn.functional.cross_entropy(
        semantic_logits.unsqueeze(0),
        target_semantic.unsqueeze(0).long(),
        reduction="none",
    )[observed_mask].mean()
    semantic_term = semantic_loss / torch.log(torch.tensor(semantic_logits.shape[0], device=semantic_logits.device, dtype=semantic_logits.dtype))

    total = (
        weights["color"] * color_term
        + weights["depth"] * depth_term
        + weights["semantic"] * semantic_term
    )
    return total, {"color": color_term, "depth": depth_term, "semantic": semantic_term}
