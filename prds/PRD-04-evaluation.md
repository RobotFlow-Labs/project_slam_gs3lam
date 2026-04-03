# PRD-04: Evaluation

> Module: SLAM-GS3LAM | Priority: P0
> Depends on: PRD-03
> Status: ✅ Complete

## Objective
Evaluation code reproduces the paper’s rendering, tracking, runtime, and semantic-reconstruction metrics and emits a gap report against Tables 1-4, 6-7, and 9.

## Context (from paper)
GS3LAM reports PSNR, SSIM, LPIPS, ATE RMSE, mIoU, and runtime/FPS on Replica and ScanNet, plus semantic comparisons on Replica.
**Paper reference**: §4.1.2, §4.2-§4.5, Tables 1-4, App C.2-C.3, Tables 6-9

## Acceptance Criteria
- [x] Replica averages can be computed for PSNR, SSIM, LPIPS, ATE, and mIoU
- [x] ScanNet averages can be computed for PSNR, SSIM, LPIPS, and ATE
- [x] Runtime report includes per-iteration, per-frame, and FPS metrics
- [x] Evaluation report prints paper value, reproduced value, and delta
- [x] Test: `uv run pytest tests/test_metrics.py tests/test_eval_reports.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_gs3lam/eval/rendering.py` | PSNR/SSIM/LPIPS aggregation | Tables 1-2 | ~180 |
| `src/anima_slam_gs3lam/eval/tracking.py` | ATE RMSE report | Table 3, 9 | ~120 |
| `src/anima_slam_gs3lam/eval/semantics.py` | mIoU aggregation | Table 4 | ~110 |
| `src/anima_slam_gs3lam/eval/runtime.py` | FPS and timing reports | Tables 6-7 | ~110 |
| `src/anima_slam_gs3lam/eval/report.py` | paper-vs-repro markdown summary | §4 | ~140 |
| `tests/test_metrics.py` | metric correctness | — | ~120 |
| `tests/test_eval_reports.py` | report formatting and targets | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- `rendered_rgb`: `Tensor[3,H,W]`
- `gt_rgb`: `Tensor[3,H,W]`
- `rendered_depth`: `Tensor[1,H,W]`
- `gt_depth`: `Tensor[1,H,W]`
- `pred_semantic`: `Tensor[H,W]`
- `gt_semantic`: `Tensor[H,W]`
- `estimated_poses`: `Tensor[T,4,4]`
- `gt_poses`: `Tensor[T,4,4]`

### Outputs
- `ReplicaReport`
- `ScanNetReport`
- `GapSummary.md`

### Algorithm
```python
# Paper §4.1.2 and Tables 1-4, 6-7, 9
report = {
    "psnr": mean(psnr_per_frame),
    "ssim": mean(ssim_per_frame),
    "lpips": mean(lpips_per_frame),
    "ate_cm": ate_rmse_meters * 100.0,
    "miou": mean_iou_percent,
    "fps": total_frames / total_time,
}
```

## Dependencies
```toml
torchmetrics = ">=1.4"
pytorch-msssim = ">=1.0"
matplotlib = ">=3.9"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| Replica benchmark scenes | 8 scenes | `/mnt/forge-data/datasets/slam/gs3lam/Replica/` | see `ASSETS.md` |
| ScanNet benchmark scenes | 6 scenes | `/mnt/forge-data/datasets/slam/gs3lam/scannet/` | see `ASSETS.md` |

## Test Plan
```bash
uv run pytest tests/test_metrics.py tests/test_eval_reports.py -v
uv run python -m anima_slam_gs3lam.eval.report --paper-targets ASSETS.md
```

## References
- Paper: §4, Tables 1-4, 6-9
- Reference impl: `repositories/GS3LAM/src/Evaluater.py`
- Depends on: PRD-03
- Feeds into: PRD-07
