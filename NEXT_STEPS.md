# SLAM-GS3LAM — Execution Ledger

Resume rule: read this file completely before writing code.
This module covers exactly one paper: `GS3LAM: Gaussian Semantic Splatting SLAM`.

## 1. Working Rules
- Work only inside `project_slam_gs3lam/`
- Use `uv` for environment management and commands
- Keep Python pinned to `3.11` for module development
- Prefer paper-faithful defaults when the paper and repo differ
- Treat `repositories/GS3LAM/` as reference code, not production module code
- Keep CUDA-first dependencies available for later Linux/GPU execution, even when developing on Mac

## 2. Paper
- **Title**: GS3LAM: Gaussian Semantic Splatting SLAM
- **ArXiv**: `2603.27781`
- **Link**: https://arxiv.org/abs/2603.27781
- **DOI**: https://doi.org/10.1145/3664647.3680739
- **Repo**: https://github.com/lif314/GS3LAM
- **Correct local PDF**: `papers/2603.27781_GS3LAM.pdf`
- **Invalid local PDF to ignore**: `papers/2503.15909_GS3LAM.pdf`
- **Verification status**: identity ✅ | repo inspected ✅ | planning aligned ✅ | CUDA training ✅ | runtime reproduction ⬜

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: CUDA training running on GPU 2 (office2, 2000 frames)
- **MVP Readiness**: 75%
- **Accomplished**:
  1. All PRD-01 through PRD-07 code complete
  2. Code review: fixed 6 critical bugs (no optimizer, pose convention, device handling, depth shape)
  3. Built gaussian-semantic-rasterization CUDA extension (sm_89 for L4)
  4. Shared CUDA rasterizer at `/mnt/forge-data/modules/03_wave7/shared_slam_cuda/`
  5. CUDA smoke test passed: 5 frames, PSNR=16.45, ATE=3.54cm
  6. Full training launched: office2, 2000 frames, nohup+disown, PID in train.pid
  7. 62/62 tests pass, ruff clean
- **Training details**:
  - Scene: office2 (complete 4x4 traj, 2000 frames)
  - GPU: CUDA_VISIBLE_DEVICES=2, NVIDIA L4 23GB
  - Config: 40 tracking iters, 60 mapping iters (paper defaults)
  - Log: `/mnt/artifacts-datai/logs/project_slam_gs3lam/train_office2_20260403_1311.log`
  - Checkpoints: `/mnt/artifacts-datai/checkpoints/project_slam_gs3lam/office2/`
- **Immediate Next Tasks**:
  1. Monitor training completion
  2. Run remaining scenes (room0-2, office0-1, office3-4) — need traj format handling
  3. Export: pth → safetensors → ONNX → TRT fp16 → TRT fp32
  4. Push to HuggingFace: ilessio-aiflowlab/project_slam_gs3lam
  5. Run /anima-hf-strategy
- **Blockers**:
  1. room0-2, office0-1 have TUM quaternion traj (500 lines) — loader now handles both formats with interpolation but accuracy may suffer vs native 4x4 format
  2. ScanNet data not available on server
  3. TUM-DEVA pseudo labels not available

## 4. Data / Weights Preflight
- **Environment detected**: `GPU_SERVER`
- **Datasets present**:
  - GS3LAM Replica (processed): `/mnt/forge-data/datasets/slam/gs3lam/Replica/` — 8 scenes, 2000 frames each ✅
  - Replica SLAM (rendered): `/mnt/forge-data/datasets/replica_slam/` — 6 scenes, 250 frames/agent ✅ (no semantics)
  - Shared symlink: `/mnt/forge-data/datasets/replica_rgbd/` → gs3lam Replica data
- **Weights present**: none needed (trained from scratch per-scene)
- **CUDA rasterizer**: built and installed ✅
  - Shared at: `/mnt/forge-data/modules/03_wave7/shared_slam_cuda/gaussian_semantic_rasterization/`

## 5. Critical Bugs Fixed (2026-04-03)
| Bug | File | Fix |
|-----|------|-----|
| No optimizer/backward in SLAM loop | pipeline/slam_loop.py | Added tracking optimizer (pose quat+trans), mapping optimizer (field+decoder) with per-param LR |
| Pose convention conflict | rendering/rasterizer.py | Fixed: compute w2c from c2w pose, correct cam_center extraction |
| Fallback renderer wrong transform | rendering/rasterizer.py | Use w2c (not c2w) for world→camera projection |
| CUDA depth/opacity unsqueeze double | rendering/rasterizer.py | Check ndim before unsqueeze — CUDA rasterizer already returns [1,H,W] |
| FrameBatch no .to(device) | types.py | Added .to() method returning new FrameBatch |
| Decoder/field not on CUDA | pipeline/slam_loop.py | Device param in GS3LAMLoop, .to(device) in bootstrap |
| Semantic glob matches vis_ files | datasets/replica.py | Filter out vis_sem_class_*.png |
| Traj format mismatch | datasets/replica.py | Support both 4x4 matrix and TUM quaternion formats with interpolation |

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Initial scaffold created |
| 2026-04-03 | Codex | Added PRD suite, tasks, corrected paper identity, vendored reference repo |
| 2026-04-03 | Codex | Completed PRD-01 through PRD-06, API, ROS2, Docker scaffolding |
| 2026-04-03 | Codex | Added Docker serve infra, training script, export pipeline, release checks |
| 2026-04-03 | Opus | Fixed 6 critical bugs, built CUDA rasterizer, started GPU training |
