# SLAM-GS3LAM

## Paper
**GS3LAM: Gaussian Semantic SLAM**
arXiv: https://arxiv.org/abs/2503.15909

## Module Identity
- Codename: SLAM-GS3LAM
- Domain: SLAM
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_slam_gs3lam/
├── pyproject.toml
├── configs/
├── src/anima_slam_gs3lam/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── AGENTS.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [SLAM-GS3LAM]
