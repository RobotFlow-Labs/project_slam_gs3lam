"""Tracking utilities for frame-to-model pose initialization and masking."""

from __future__ import annotations

import torch


def initialize_pose(prev_pose: torch.Tensor, prev_prev_pose: torch.Tensor | None = None) -> torch.Tensor:
    """Constant-velocity pose initialization from Eq. (19)."""

    if prev_prev_pose is None:
        return prev_pose.clone()
    return prev_pose @ torch.linalg.inv(prev_prev_pose) @ prev_pose


def compute_observed_mask(
    opacity: torch.Tensor,
    depth_render: torch.Tensor,
    depth_gt: torch.Tensor,
    *,
    tau_obs: float = 0.99,
    depth_scale: float = 10.0,
) -> torch.Tensor:
    """Observed-region mask used in tracking loss from Eq. (21)."""

    depth_error = torch.abs(depth_gt - depth_render) * (depth_gt > 0)
    valid_error = depth_error[depth_gt > 0]
    median = valid_error.median() if valid_error.numel() else torch.tensor(0.0, device=depth_gt.device, dtype=depth_gt.dtype)
    depth_consistent = depth_error < (depth_scale * median.clamp_min(1e-6))
    return (opacity > tau_obs) & depth_consistent & (depth_gt > 0)


def pose_to_components(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pose.shape != (4, 4):
        raise ValueError("pose must have shape [4, 4]")
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return rotation, translation
