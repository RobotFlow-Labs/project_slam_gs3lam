# SLAM-GS3LAM: GS3LAM: Gaussian Semantic Splatting SLAM — Implementation PRD
## ANIMA Wave-7 Module

**Status:** PRD Suite Generated  
**Version:** 0.2  
**Date:** 2026-04-03  
**Correct Paper:** GS3LAM: Gaussian Semantic Splatting SLAM  
**Correct Paper Link:** https://arxiv.org/abs/2603.27781  
**ACM DOI:** https://doi.org/10.1145/3664647.3680739  
**Reference Repo:** https://github.com/lif314/GS3LAM  
**Compute:** GPU-NEED  
**Functional Name:** `slam-gs3lam`  
**Target Package:** `src/anima_slam_gs3lam/`

## 1. Executive Summary
GS3LAM is a dense semantic RGB-D SLAM system built around a Semantic Gaussian Field (SG-Field). The paper’s core method uses 3D semantic Gaussians plus differentiable splatting to jointly optimize camera poses, geometry, appearance, and semantics in real time. The ANIMA implementation plan keeps that structure intact: low-dimensional semantic features per Gaussian, a lightweight decoder for per-pixel semantics, adaptive Gaussian expansion from unobserved regions, Depth-adaptive Scale Regularization (DSR), and Random Sampling-based Keyframe Mapping (RSKM). The repo now contains a paper-faithful PRD suite and task decomposition for rebuilding the method in this module.

## 2. Paper Verification Status
- [x] Correct arXiv paper identified: `2603.27781` on March 29, 2026
- [x] ACM MM 2024 DOI confirmed
- [x] GitHub repo confirmed and inspected in `repositories/GS3LAM/`
- [x] Correct paper PDF stored locally at `papers/2603.27781_GS3LAM.pdf`
- [x] Local PDF mismatch detected and documented for `papers/2503.15909_GS3LAM.pdf`
- [ ] Reference repo executed locally on benchmark data
- [ ] Datasets downloaded and validated on shared storage
- [ ] Metrics reproduced within acceptance thresholds
- **Verdict:** VERIFIED FOR PLANNING, NOT YET VERIFIED FOR EXECUTION

## 3. What We Take From The Paper
- Semantic Gaussian Field representation with per-Gaussian position, covariance, opacity, RGB color, and low-dimensional semantic feature.
- Differentiable splatting for RGB, depth, and semantic feature rendering.
- Decoder path from low-dimensional semantic features to per-class semantic logits.
- Decoupled optimization: frame-to-model tracking with frozen field, followed by mapping with frozen poses.
- Adaptive Gaussian expansion using cumulative opacity and depth inconsistency masks.
- DSR to keep Gaussian scales in a depth-consistent range and reduce semantic/geometry misalignment.
- RSKM to mitigate optimization bias and forgetting in incremental mapping.
- Benchmark targets and runtime envelopes from Replica and ScanNet.

## 4. What We Skip
- Any redesign that replaces splatting with NeRF ray marching or another implicit renderer.
- Any attempt to remove semantics from the core representation.
- TUM as a primary quantitative benchmark, since the paper uses it only with pseudo labels and not as the main evaluation suite.
- Premature MLX-native parity for training; paper-faithful reproduction stays CUDA-first.

## 5. What We Adapt
- ANIMA package conventions: `uv`, typed configs, tests, modular package layout, service/API boundaries, and ROS2 integration.
- Stable API and ROS2 wrappers for downstream robotics use.
- Production diagnostics, release checks, and artifact export, which are not part of the paper.
- Explicit reconciliation of paper-vs-repo drift, especially ScanNet iteration counts.

## 6. Architecture
### Inputs
- RGB frame: `Tensor[3,H,W]`
- Depth frame: `Tensor[1,H,W]`
- Semantic labels: `Tensor[H,W]`
- Camera intrinsics: `Tensor[3,3]`
- Pose: `Tensor[4,4]`

### Core State
- SG-Field Gaussians:
  - `means3d: Tensor[N,3]`
  - `rotations: Tensor[N,4]`
  - `log_scales: Tensor[N,1|3]`
  - `opacities: Tensor[N,1]`
  - `rgb: Tensor[N,3]`
  - `semantic_features: Tensor[N,16]`
- Semantic decoder logits: `Tensor[256,H,W]`

### Outputs
- Estimated poses
- Rendered RGB/depth/semantic maps
- Checkpoints and evaluation reports
- Service and ROS2 adapters

## 7. Implementation Phases
### Phase 1 — Foundation + Paper-Faithful Core
- Correct stale scaffold naming and package layout
- Implement dataset/config contracts
- Build SG-Field, semantic decoder, rasterizer wrapper, and DSR

### Phase 2 — Online SLAM Reproduction
- Implement tracking, mapping, adaptive expansion, and RSKM
- Run Replica/ScanNet benchmark scenes
- Save checkpoints and intermediate visualizations

### Phase 3 — Evaluation + Reproduction
- Reproduce Tables 1-4 and 9 metrics
- Produce runtime/FPS reports aligned with Tables 6-7
- Emit paper-vs-repro gap reports

### Phase 4 — ANIMA Integration
- Expose FastAPI service
- Add Docker runtime
- Add ROS2 node and launch support

### Phase 5 — Productionization
- Add export, regression gates, and release checklist
- Validate demo readiness against quantitative thresholds

## 8. Datasets
| Dataset | Scenes | Source | Phase Needed |
|---|---|---|---|
| Replica semantic | `room0`, `room1`, `room2`, `office0-4` | HF `3David14/GS3LAM-Replica` | Phase 1 |
| ScanNet | `scene0000_00`, `scene0059_00`, `scene0106_00`, `scene0169_00`, `scene0181_00`, `scene0207_00` | ScanNet + frame extraction | Phase 2 |
| TUM-DEVA | `freiburg1_desk` | HF `3David14/TUM-DEVA` | Phase 3 |

## 9. Dependencies on Other Wave Projects
| Needs output from | What it provides |
|---|---|
| None required for paper-faithful reproduction | — |
| Upstream perception node in deployed stack | real-time semantic labels for live ROS2 use |

## 10. Success Criteria
- Replica avg PSNR `>= 35.0 dB`
- Replica avg SSIM `>= 0.985`
- Replica avg LPIPS `<= 0.065`
- Replica avg mIoU `>= 95.0%`
- Replica avg ATE `<= 0.50 cm`
- ScanNet avg PSNR `>= 21.5 dB`
- ScanNet avg ATE `<= 12.5 cm`
- Replica rendering speed `>= 90 FPS`
- Service and ROS2 smoke tests pass

## 11. Risk Assessment
- The current repo scaffold is stale and will mislead implementation unless corrected first.
- The official implementation is CUDA-specific and uses older PyTorch/CUDA pins.
- The paper and repo disagree on some ScanNet iteration settings.
- Benchmark datasets are not yet present locally.
- Semantic labels for live deployment are an external dependency.

## 12. Build Plan
| PRD# | Task | Status |
|---|---|---|
| [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | ⬜ |
| [PRD-02](prds/PRD-02-core-model.md) | Core Model | ⬜ |
| [PRD-03](prds/PRD-03-inference.md) | Inference | ⬜ |
| [PRD-04](prds/PRD-04-evaluation.md) | Evaluation | ⬜ |
| [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | ⬜ |
| [PRD-06](prds/PRD-06-ros2-integration.md) | ROS2 Integration | ⬜ |
| [PRD-07](prds/PRD-07-production.md) | Production | ⬜ |

## 13. Task Breakdown
- See `tasks/PRD-0101.md` through `tasks/PRD-0703.md`

## 14. Shenzhen Demo Target
- Minimum: Phase 3 with one benchmark reproduction and API smoke test
- Preferred: Phase 4 with ROS2 replay demo on RGB-D + semantic stream
