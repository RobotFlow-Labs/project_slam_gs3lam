# SLAM-GS3LAM — Execution Ledger

Resume rule: read this file completely before writing code.
This module covers exactly one paper: `GS3LAM: Gaussian Semantic Splatting SLAM`.

## 1. Working Rules
- Work only inside `project_slam_gs3lam/`
- Use `uv` for environment management and commands
- Keep Python pinned to `3.11` for module development
- Prefer paper-faithful defaults when the paper and repo differ
- Treat `repositories/GS3LAM/` as reference code, not production module code

## 2. Paper
- **Title**: GS3LAM: Gaussian Semantic Splatting SLAM
- **ArXiv**: `2603.27781`
- **Link**: https://arxiv.org/abs/2603.27781
- **DOI**: https://doi.org/10.1145/3664647.3680739
- **Repo**: https://github.com/lif314/GS3LAM
- **Correct local PDF**: `papers/2603.27781_GS3LAM.pdf`

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: CUDA training running on GPU 2 (office2, 2000 frames)
- **MVP Readiness**: 78%
- **Training**:
  - PID: see `/mnt/artifacts-datai/logs/project_slam_gs3lam/train.pid`
  - Scene: office2, 2000 frames, 40 tracking + 60 mapping iters
  - GPU: CUDA_VISIBLE_DEVICES=2, NVIDIA L4 23GB
  - Caps: 100K init Gaussians, 10K expansion/frame, prune every 50 frames
  - Log: latest in `/mnt/artifacts-datai/logs/project_slam_gs3lam/train_office2_*.log`
- **Done**:
  - [x] PRD-01 through PRD-07 code complete
  - [x] Fixed 8 critical bugs (optimizer, pose, device, depth shape, Gaussian caps, pruning)
  - [x] Built CUDA rasterizer (sm_89 for L4) — shared at `/mnt/forge-data/modules/03_wave7/shared_slam_cuda/`
  - [x] TensorRT fp16+fp32 export pipeline
  - [x] ANIMA standard endpoints (/health, /ready, /info, /predict)
  - [x] Dockerfile.serve with CUDA rasterizer build
  - [x] __main__.py for python -m anima_slam_gs3lam
  - [x] 62/62 tests pass, ruff clean
  - [x] Pushed to GitHub
- **TODO when training completes**:
  - [ ] Check final metrics vs paper targets
  - [ ] Run export pipeline: pth → safetensors → ONNX → TRT fp16 → TRT fp32
  - [ ] Run /anima-hf-strategy
  - [ ] Run additional scenes (room0, room1, room2, office0, office1)
- **Blockers**:
  - room0-2, office0-1 traj files are TUM quaternion format (500 lines) — interpolation to 2000 may degrade pose accuracy
  - ScanNet / TUM-DEVA datasets not on server

## 4. Data
- **GS3LAM Replica** (processed): `/mnt/forge-data/datasets/slam/gs3lam/Replica/` — 8 scenes, 2000 frames each ✅
- **Replica SLAM** (rendered by W7_COKO): `/mnt/forge-data/datasets/replica_slam/` — 6 scenes, 250 frames/agent (no semantics)
- **Shared symlink**: `/mnt/forge-data/datasets/replica_rgbd/` → gs3lam Replica
- **CUDA rasterizer**: `/mnt/forge-data/modules/03_wave7/shared_slam_cuda/gaussian_semantic_rasterization/`

## 5. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research | Initial scaffold |
| 2026-04-03 | Codex | PRD-01 through PRD-06, API, ROS2, Docker |
| 2026-04-03 | Codex | Docker serve infra, training script, export, release checks |
| 2026-04-03 | Opus | Fixed 8 bugs, built CUDA rasterizer, added ANIMA infra, started training |
