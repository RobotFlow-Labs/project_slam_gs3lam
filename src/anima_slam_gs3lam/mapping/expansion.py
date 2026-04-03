"""Adaptive Gaussian expansion utilities."""

from __future__ import annotations

import torch

from anima_slam_gs3lam.sg_field import SemanticGaussianField
from anima_slam_gs3lam.types import FrameBatch


def compute_unobserved_mask(
    opacity: torch.Tensor,
    depth_render: torch.Tensor,
    depth_gt: torch.Tensor,
    *,
    tau_unobs: float = 0.1,
    depth_scale: float = 50.0,
) -> torch.Tensor:
    """Unobserved-region mask from Eq. (11)."""

    depth_error = torch.abs(depth_gt - depth_render) * (depth_gt > 0)
    valid_error = depth_error[depth_gt > 0]
    median = valid_error.median() if valid_error.numel() else torch.tensor(0.0, device=depth_gt.device, dtype=depth_gt.dtype)
    depth_violation = (depth_render > depth_gt) & (depth_error > depth_scale * median.clamp_min(1e-6))
    return (opacity < tau_unobs) | depth_violation


def frame_to_world_points(frame: FrameBatch, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift masked RGB-D pixels into world-space points."""

    if mask.ndim != 2:
        raise ValueError("mask must have shape [H, W]")
    depth = frame.depth[0]
    valid = mask & (depth > 0)
    indices = torch.nonzero(valid, as_tuple=False)
    if indices.numel() == 0:
        return (
            torch.empty((0, 3), dtype=frame.rgb.dtype, device=frame.rgb.device),
            torch.empty((0, 3), dtype=frame.rgb.dtype, device=frame.rgb.device),
        )

    rows = indices[:, 0].to(dtype=frame.rgb.dtype)
    cols = indices[:, 1].to(dtype=frame.rgb.dtype)
    z = depth[indices[:, 0], indices[:, 1]]
    fx = frame.intrinsics[0, 0]
    fy = frame.intrinsics[1, 1]
    cx = frame.intrinsics[0, 2]
    cy = frame.intrinsics[1, 2]

    x = (cols - cx) / fx * z
    y = (rows - cy) / fy * z
    cam_points = torch.stack([x, y, z], dim=-1)
    ones = torch.ones((cam_points.shape[0], 1), dtype=cam_points.dtype, device=cam_points.device)
    world_points = (frame.pose @ torch.cat([cam_points, ones], dim=-1).T).T[:, :3]
    colors = frame.rgb[:, indices[:, 0], indices[:, 1]].T
    return world_points, colors


def estimate_mean_sq_dist(points_camera: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    scale_gaussian = points_camera[:, 2].clamp_min(1e-6) / ((fx + fy) / 2.0)
    return scale_gaussian.square()


def expand_field_from_frame(
    field: SemanticGaussianField,
    frame: FrameBatch,
    unobserved_mask: torch.Tensor,
    *,
    max_new_gaussians: int = 1000,
) -> int:
    """Append new Gaussians from unobserved RGB-D pixels.

    Caps new Gaussians per frame to ``max_new_gaussians`` via uniform subsampling
    to prevent unbounded memory growth.
    """
    world_points, colors = frame_to_world_points(frame, unobserved_mask)
    if world_points.shape[0] == 0:
        return 0

    # Subsample if too many new points
    n = world_points.shape[0]
    if n > max_new_gaussians:
        indices = torch.randperm(n, device=world_points.device)[:max_new_gaussians]
        world_points = world_points[indices]
        colors = colors[indices]

    camera_points = torch.linalg.solve(
        frame.pose,
        torch.cat([world_points, torch.ones(world_points.shape[0], 1, device=world_points.device, dtype=world_points.dtype)], dim=-1).T,
    ).T[:, :3]
    mean_sq_dist = estimate_mean_sq_dist(camera_points, frame.intrinsics)
    field.append_gaussians(world_points, colors, mean_sq_dist=mean_sq_dist)
    return int(world_points.shape[0])
