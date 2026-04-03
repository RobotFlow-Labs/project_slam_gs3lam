"""Tracking metrics."""

from __future__ import annotations

import torch


def ate_rmse_cm(estimated_poses: torch.Tensor, gt_poses: torch.Tensor) -> float:
    if estimated_poses.shape != gt_poses.shape:
        raise ValueError("estimated_poses and gt_poses must have the same shape")
    est_trans = estimated_poses[:, :3, 3]
    gt_trans = gt_poses[:, :3, 3]
    rmse_m = torch.sqrt(torch.mean(torch.sum((est_trans - gt_trans) ** 2, dim=-1)))
    return float(rmse_m.item() * 100.0)
