# PRD-05: API & Docker

> Module: SLAM-GS3LAM | Priority: P1
> Depends on: PRD-03
> Status: 🟡 In progress

## Objective
GS3LAM is callable as a service with a reproducible containerized runtime that can ingest RGB-D-semantics frames and return poses plus rendered outputs.

## Context (from paper)
The paper is offline/research oriented, but ANIMA requires a service boundary that preserves the paper’s online frame-to-model loop while exposing it through a stable API.
**Paper reference**: §3.1-§3.4 for runtime behavior; App C.2 for timing expectations

## Acceptance Criteria
- [x] FastAPI app exposes health, load-scene, step-frame, and snapshot endpoints
- [x] Request/response schemas include pose, RGB render, depth render, semantic logits, and timing
- [ ] CUDA-enabled Docker image builds and launches the service
- [ ] Smoke test can run a 5-frame sequence in-container
- [x] Test: `uv run pytest tests/test_api.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_gs3lam/api/app.py` | FastAPI entrypoint | runtime adaptation | ~140 |
| `src/anima_slam_gs3lam/api/schemas.py` | request/response contracts | runtime adaptation | ~120 |
| `src/anima_slam_gs3lam/api/service.py` | SLAM session manager | §3.4 | ~180 |
| `docker/Dockerfile.cuda` | CUDA runtime image | App C.2 | ~80 |
| `docker/docker-compose.api.yml` | local service orchestration | — | ~50 |
| `tests/test_api.py` | API smoke tests | — | ~100 |

## Architecture Detail (from paper)
### Inputs
- `rgb_png` or serialized `Tensor[3,H,W]`
- `depth_png` or serialized `Tensor[1,H,W]`
- `semantic_png` or serialized `Tensor[H,W]`
- `intrinsics`
- `frame_id`

### Outputs
- `pose_w2c: Tensor[4,4]`
- `render_rgb: Tensor[3,H,W]`
- `render_depth: Tensor[1,H,W]`
- `semantic_logits: Tensor[256,H,W]`
- `timings_ms`

### Algorithm
```python
@app.post("/v1/gs3lam/step")
def step_frame(req: StepFrameRequest) -> StepFrameResponse:
    state = service.step(req)
    return StepFrameResponse.from_state(state)
```

## Dependencies
```toml
fastapi = ">=0.115"
uvicorn = ">=0.30"
orjson = ">=3.10"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| test scene fragment | 5-10 frames | `/mnt/forge-data/datasets/slam/gs3lam/smoke/` | derived from benchmark datasets |

## Test Plan
```bash
uv run pytest tests/test_api.py -v
docker compose -f docker/docker-compose.api.yml up --build
```

## References
- Paper: §3.1-§3.4, App C.2
- Depends on: PRD-03
- Feeds into: PRD-06, PRD-07
