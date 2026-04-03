import torch

from anima_slam_gs3lam.eval.rendering import peak_signal_noise_ratio, rendering_metrics


def test_psnr_monotonicity():
    gt = torch.ones(3, 8, 8)
    close = gt * 0.9
    far = torch.zeros(3, 8, 8)
    assert peak_signal_noise_ratio(gt, close) > peak_signal_noise_ratio(gt, far)


def test_rendering_metrics_return_depth_error():
    metrics = rendering_metrics(
        pred_rgb=torch.zeros(3, 8, 8),
        gt_rgb=torch.ones(3, 8, 8),
        pred_depth=torch.ones(1, 8, 8),
        gt_depth=torch.ones(1, 8, 8),
    )
    assert metrics.depth_mae == 0.0
    assert metrics.lpips > 0.0
