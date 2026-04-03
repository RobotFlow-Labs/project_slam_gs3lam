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
- **Verification status**: identity ✅ | repo inspected ✅ | planning aligned ✅ | runtime reproduction ⬜

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: PRD-01 through PRD-04 complete, PRD-05 and PRD-06 in local verification state
- **MVP Readiness**: 68%
- **Accomplished**:
  1. Replaced stale `TSUKUYOMI` package metadata with `anima_slam_gs3lam`
  2. Updated `pyproject.toml` for Python `3.11` and added Mac/CUDA-aware dependency groups
  3. Added typed GS3LAM config surface with paper defaults and explicit repo overrides
  4. Ported Replica, ScanNet, and TUM dataset contracts/loaders into the module package
  5. Implemented SG-Field, semantic decoder, rasterizer wrapper, DSR utilities, and tracking/mapping loop
  6. Added evaluation metrics, paper-target gap reporting, FastAPI service endpoints, and CUDA-ready Docker scaffolding
  7. Added ROS2 topic contracts, config, node shim, and launch scaffolding without requiring `rclpy` on Mac
  8. Verified `uv sync --python 3.11 --extra dev`
  9. Verified `uv run --python 3.11 pytest -q`
  10. Verified `uv run --python 3.11 pytest tests/test_api.py -v`
  11. Verified `uv run --python 3.11 pytest tests/test_ros2_contracts.py -v`
  12. Verified `uv run --python 3.11 python -m anima_slam_gs3lam.eval.report --paper-targets ASSETS.md`
  13. Verified `uv run --python 3.11 ruff check src/ tests/ scripts/ launch/`
- **Immediate Next Tasks**:
  1. Verify `docker/Dockerfile.cuda` and `docker/docker-compose.api.yml` on a Linux/CUDA host
  2. Verify `launch/gs3lam.launch.py` and the ROS2 node on a real ROS2 host with `rclpy`
  3. Add `anima_module.yaml` for ANIMA-native module registration
  4. Add export/regression gates from PRD-07
- **Blockers**:
  1. Benchmark datasets are not present locally or on shared storage
  2. CUDA rasterizer cannot be compiled on this Mac; Linux/CUDA path must remain optional until GPU server execution
  3. No pretrained checkpoints are available yet

## 4. Data / Weights Preflight
- **Environment detected**: `MAC_LOCAL`
- **Datasets present**: none
- **Weights present**: none
- **Reference repo present**: `repositories/GS3LAM/`
- **ANIMA infra present**:
  - `pyproject.toml` ✅
  - `configs/default.toml` ✅
  - `tests/` ✅
  - `anima_module.yaml` ⬜
  - `Dockerfile.serve` ⬜
  - `docker-compose.serve.yml` ⬜
  - `docker/Dockerfile.cuda` ✅
  - `docker/docker-compose.api.yml` ✅
  - `src/anima_slam_gs3lam/serve.py` ✅

## 5. Hardware Notes
- Mac Studio / Apple Silicon: active local development target
- CUDA / Linux server: required for paper-faithful rasterizer execution and training
- Keep dual-path design:
  - local path: package import, config, tests, CPU/MPS-safe code
  - server path: CUDA extension build, training, benchmarking

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Initial scaffold created |
| 2026-04-03 | Codex | Added PRD suite, tasks, corrected paper identity, vendored reference repo |
| 2026-04-03 | Codex | Completed PRD-01: package rename, typed config, dataset loaders, Python 3.11 `uv` verification |
| 2026-04-03 | Codex | Completed PRD-02 and PRD-03: SG-Field core, rasterizer wrapper, tracking/mapping loop, Replica smoke runner |
| 2026-04-03 | Codex | Completed PRD-04 and local PRD-05 API slice: evaluation reports, FastAPI service layer, Docker scaffolding |
| 2026-04-03 | Codex | Completed local PRD-06 contracts: ROS2 topic helpers, node shim, config, and launch scaffold |
