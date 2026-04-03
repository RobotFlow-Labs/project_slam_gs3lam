import torch

from anima_slam_gs3lam.eval.report import build_gap_report, parse_paper_targets
from anima_slam_gs3lam.eval.semantics import mean_iou_percent
from anima_slam_gs3lam.eval.tracking import ate_rmse_cm


def test_ate_rmse_cm():
    gt = torch.eye(4).repeat(2, 1, 1)
    est = gt.clone()
    est[1, 0, 3] = 1.0
    assert ate_rmse_cm(est, gt) > 0.0


def test_mean_iou_percent():
    pred = torch.tensor([[0, 1], [1, 1]])
    gt = torch.tensor([[0, 1], [0, 1]])
    assert mean_iou_percent(pred, gt) > 0.0


def test_gap_report_contains_targets():
    targets = parse_paper_targets("ASSETS.md")
    report = build_gap_report({"replica_psnr": 35.0}, targets)
    assert "replica_psnr" in report
    assert "35.0000" in report
