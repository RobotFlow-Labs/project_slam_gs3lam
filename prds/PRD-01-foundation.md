# PRD-01: Foundation & Config

> Module: SLAM-GS3LAM | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
The module scaffold, configuration system, and dataset contracts are corrected and aligned with the GS3LAM paper and reference repo.

## Context (from paper)
GS3LAM processes RGB, depth, and semantic labels jointly and relies on dataset-specific camera geometry, incremental frame access, and reproducible hyperparameters.
**Paper reference**: §3.1 "Framework Overview", §4.1 "Setup", App C.1 "Further Implementation Details"

## Acceptance Criteria
- [ ] Project metadata is renamed from stale `TSUKUYOMI` placeholders to `SLAM-GS3LAM`
- [ ] `src/anima_slam_gs3lam/` replaces the stale package path for new implementation work
- [ ] Replica, ScanNet, and TUM dataset registries expose `(color, depth, intrinsics, pose, semantics)`
- [ ] Config schema captures paper-faithful defaults and repo-specific overrides separately
- [ ] Test: `uv run pytest tests/test_config.py tests/test_datasets.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_gs3lam/config.py` | Typed runtime configuration and dataset presets | §4.1, App C.1 | ~180 |
| `src/anima_slam_gs3lam/types.py` | Shared tensor/data contracts | §3.1 | ~100 |
| `src/anima_slam_gs3lam/datasets/registry.py` | Dataset selection by name | §4.1.2 | ~60 |
| `src/anima_slam_gs3lam/datasets/replica.py` | Replica semantic loader | §4.1.2 | ~180 |
| `src/anima_slam_gs3lam/datasets/scannet.py` | ScanNet semantic loader | §4.1.2 | ~180 |
| `src/anima_slam_gs3lam/datasets/tum.py` | TUM + pseudo semantics loader | README datasets | ~180 |
| `tests/test_config.py` | Config coverage | — | ~120 |
| `tests/test_datasets.py` | Loader contracts and shape tests | — | ~160 |

## Architecture Detail (from paper)
### Inputs
- `rgb`: `Tensor[3,H,W]`
- `depth`: `Tensor[1,H,W]`
- `semantic`: `Tensor[H,W]`
- `intrinsics`: `Tensor[3,3]`
- `pose`: `Tensor[4,4]`

### Outputs
- `FrameBatch`: typed object holding RGB-D-semantics plus metadata
- `GS3LAMConfig`: validated settings with per-dataset defaults

### Algorithm
```python
# Paper §4.1.2 + repo datasets/*
class FrameBatch(NamedTuple):
    rgb: torch.Tensor
    depth: torch.Tensor
    semantic: torch.Tensor
    intrinsics: torch.Tensor
    pose: torch.Tensor
    frame_id: int


class DatasetRegistry:
    def build(self, name: str, cfg: DatasetConfig) -> Dataset:
        ...
```

## Dependencies
```toml
pydantic = ">=2.8"
pyyaml = ">=6.0"
natsort = ">=8.4"
pillow = ">=10.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| Replica semantic dataset | 8 scenes | `/mnt/forge-data/datasets/slam/gs3lam/Replica/` | HF dataset in `ASSETS.md` |
| ScanNet extracted frames | 6 scenes | `/mnt/forge-data/datasets/slam/gs3lam/scannet/` | official ScanNet + `.sens` extraction |
| TUM-DEVA | `freiburg1_desk` | `/mnt/forge-data/datasets/slam/gs3lam/TUM-DEVA/` | HF dataset in `ASSETS.md` |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_datasets.py -v
uv run ruff check src/ tests/
```

## References
- Paper: §3.1, §4.1, App C.1
- Reference impl: `repositories/GS3LAM/src/datasets/*.py`
- Feeds into: PRD-02, PRD-03
