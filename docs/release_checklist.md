# GS3LAM Release Checklist

## Pre-Release

- [ ] All 62 tests pass: `uv run pytest tests/ -v`
- [ ] Lint clean: `uv run ruff check src/ tests/ scripts/`
- [ ] Training completed on at least one Replica scene
- [ ] Checkpoint saved to `/mnt/artifacts-datai/checkpoints/project_slam_gs3lam/`
- [ ] NEXT_STEPS.md updated with final status

## Metrics Verification

Run `uv run python -m anima_slam_gs3lam.release_checks` against eval reports.

| Metric | Paper | Target | Status |
|--------|-------|--------|--------|
| Replica avg PSNR | 36.26 dB | >= 35.0 dB | |
| Replica avg SSIM | 0.989 | >= 0.985 | |
| Replica avg LPIPS | 0.052 | <= 0.065 | |
| Replica avg ATE | 0.37 cm | <= 0.50 cm | |
| Replica avg mIoU | 96.63% | >= 95.0% | |
| Replica FPS | 109.12 | >= 90 FPS | |

## Export Pipeline

All formats mandatory:

- [ ] `pth` checkpoint
- [ ] `safetensors` (field + decoder)
- [ ] `ONNX` (decoder)
- [ ] `TensorRT FP32` (decoder)
- [ ] `TensorRT FP16` (decoder)
- [ ] `npy` poses

Run: `uv run python -m anima_slam_gs3lam.export <checkpoint_path>`

## Docker

- [ ] `docker compose -f docker-compose.serve.yml --profile api build`
- [ ] `docker compose -f docker-compose.serve.yml --profile api up -d`
- [ ] `curl localhost:8080/health` returns `{"status":"ok"}`
- [ ] `curl localhost:8080/ready` returns `{"ready":true}`
- [ ] `curl localhost:8080/info` returns module info

## ROS2

- [ ] Launch file exists: `launch/gs3lam.launch.py`
- [ ] Node shim compiles: `uv run python -c "from anima_slam_gs3lam.ros2.node import GS3LAMNode"`
- [ ] Topic contracts verified: `uv run pytest tests/test_ros2_contracts.py -v`

## ANIMA Infrastructure

- [ ] `anima_module.yaml` complete
- [ ] Registry entry at `/mnt/forge-data/anima-infra-main/registry/slam_gs3lam.yaml`
- [ ] `Dockerfile.serve` builds
- [ ] `.env.serve` has correct module identity
- [ ] `__main__.py` works: `uv run python -m anima_slam_gs3lam`

## HuggingFace Push

- [ ] Repo: `ilessio-aiflowlab/project_slam_gs3lam`
- [ ] All export formats uploaded
- [ ] Model card with training config, metrics, paper reference
- [ ] Training report uploaded

## Git

- [ ] All changes committed with `[SLAM-GS3LAM]` prefix
- [ ] Pushed to `origin/main`
- [ ] NEXT_STEPS.md shows MVP >= 80%
