"""Rendering metrics and aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RenderingMetrics:
    psnr: float
    ssim: float
    lpips: float
    depth_mae: float


def peak_signal_noise_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def structural_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator.clamp_min(eps)


def lpips_proxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rendering_metrics(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
) -> RenderingMetrics:
    valid_depth = gt_depth > 0
    if torch.any(valid_depth):
        depth_mae = torch.abs(pred_depth - gt_depth)[valid_depth].mean().item()
    else:
        depth_mae = 0.0
    return RenderingMetrics(
        psnr=peak_signal_noise_ratio(pred_rgb, gt_rgb).item(),
        ssim=structural_similarity(pred_rgb, gt_rgb).item(),
        lpips=lpips_proxy(pred_rgb, gt_rgb).item(),
        depth_mae=depth_mae,
    )


def aggregate_rendering_metrics(values: list[RenderingMetrics]) -> RenderingMetrics:
    if not values:
        return RenderingMetrics(psnr=0.0, ssim=0.0, lpips=0.0, depth_mae=0.0)
    return RenderingMetrics(
        psnr=sum(item.psnr for item in values) / len(values),
        ssim=sum(item.ssim for item in values) / len(values),
        lpips=sum(item.lpips for item in values) / len(values),
        depth_mae=sum(item.depth_mae for item in values) / len(values),
    )
