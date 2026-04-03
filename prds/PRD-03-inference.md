# PRD-03: Inference

> Module: SLAM-GS3LAM | Priority: P0
> Depends on: PRD-02
> Status: ✅ Complete

## Objective
The online GS3LAM loop reproduces frame-to-model tracking, adaptive Gaussian expansion, DSR-aware mapping, and RSKM keyframe optimization.

## Context (from paper)
GS3LAM separates tracking and mapping, adds Gaussians from unobserved regions, regularizes scales with DSR, and uses RSKM instead of local covisibility-only keyframe mapping.
**Paper reference**: §3.3, §3.4, Eq. (10)-(21)

## Acceptance Criteria
- [ ] Constant-velocity pose initialization matches Eq. (19)
- [ ] `M_unobs` and `M_obs` masks match Eq. (11) and Eq. (21)
- [ ] Mapping loss matches Eq. (18); tracking loss matches Eq. (21)
- [ ] RSKM sampler matches Eq. (13)
- [ ] CLI can run a single Replica or ScanNet scene end-to-end and emit checkpoints
- [ ] Test: `uv run pytest tests/test_tracking.py tests/test_mapping.py tests/test_rskm.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_gs3lam/tracking/tracker.py` | frame-to-model pose optimization | Eq. (19)-(21) | ~180 |
| `src/anima_slam_gs3lam/mapping/expansion.py` | adaptive Gaussian expansion | Eq. (10)-(11) | ~150 |
| `src/anima_slam_gs3lam/mapping/rskm.py` | keyframe sampler | Eq. (13) | ~110 |
| `src/anima_slam_gs3lam/losses/tracking.py` | tracking objective | Eq. (21) | ~120 |
| `src/anima_slam_gs3lam/losses/mapping.py` | mapping objective | Eq. (18) | ~160 |
| `src/anima_slam_gs3lam/pipeline/slam_loop.py` | online orchestration | §3.3-§3.4 | ~240 |
| `scripts/run_replica.py` | Replica scene runner | §4.1.2 | ~80 |
| `tests/test_tracking.py` | pose update tests | — | ~120 |
| `tests/test_mapping.py` | masks and loss tests | — | ~140 |
| `tests/test_rskm.py` | sampler tests | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- `field_state`: SG-Field with `N` Gaussians
- `current_frame`: `FrameBatch`
- `keyframes`: list of `FrameBatch`
- `camera_pose_t_minus_1`, `camera_pose_t_minus_2`: `Tensor[4,4]`

### Outputs
- `updated_pose`: `Tensor[4,4]`
- `updated_field`: SG-Field
- `checkpoint`: structured state dict containing poses, Gaussians, decoder

### Algorithm
```python
# Paper §3.3-§3.4
T_t = T_t_minus_1 @ torch.linalg.inv(T_t_minus_2) @ T_t_minus_1
M_unobs = (opacity < tau_unobs) | ((depth_render > depth_gt) & (l1_depth > 50 * median_l1))
M_obs = (opacity > tau_obs) & (l1_depth < 10 * median_l1)
loss_map = 0.5 * color + 1.0 * depth + 0.01 * semantic + 0.01 * big + 0.001 * small
loss_track = M_obs * (0.5 * color + 1.0 * depth + 0.001 * semantic)
```

## Dependencies
```toml
tqdm = ">=4.66"
opencv-python = ">=4.10"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| Replica office0 scene | one RGB-D sequence | `/mnt/forge-data/datasets/slam/gs3lam/Replica/office0/` | HF dataset |
| ScanNet 0059 scene | one RGB-D sequence | `/mnt/forge-data/datasets/slam/gs3lam/scannet/scene0059_00/` | ScanNet |

## Test Plan
```bash
uv run pytest tests/test_tracking.py tests/test_mapping.py tests/test_rskm.py -v
uv run python scripts/run_replica.py --scene office0 --max-frames 10
```

## References
- Paper: §3.3, §3.4, Eq. (10)-(21)
- Reference impl: `repositories/GS3LAM/src/GS3LAM.py`, `src/Mapper.py`, `src/Tracker.py`, `src/Loss.py`, `src/GaussianManager.py`
- Depends on: PRD-02
- Feeds into: PRD-04, PRD-05, PRD-06
