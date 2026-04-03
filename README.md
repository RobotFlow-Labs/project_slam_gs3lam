# SLAM-GS3LAM — ANIMA Module

> **GS3LAM: Gaussian Semantic SLAM**
> Paper: [arXiv:2503.15909](https://arxiv.org/abs/2503.15909)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
SLAM

## Status
- [ ] Paper read + ASSETS.md created
- [ ] PRD-01 through PRD-07
- [ ] Training pipeline
- [ ] GPU training
- [ ] Export: pth + safetensors + ONNX + TRT fp16 + TRT fp32
- [ ] Push to HuggingFace
- [ ] Docker serving

## Quick Start
```bash
cd project_slam_gs3lam
uv venv .venv --python python3.11 && uv sync
uv run pytest tests/ -v
```

## License
MIT — AIFLOW LABS LIMITED
