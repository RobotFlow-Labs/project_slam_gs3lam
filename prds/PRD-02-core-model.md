# PRD-02: Core Model

> Module: SLAM-GS3LAM | Priority: P0
> Depends on: PRD-01
> Status: ✅ Complete

## Objective
The Semantic Gaussian Field, differentiable semantic splatting path, and decoder are implemented with the same data model as GS3LAM.

## Context (from paper)
GS3LAM represents the scene as a Semantic Gaussian Field and renders RGB, depth, and semantic feature maps by splatting ordered Gaussians, then decodes semantic features to semantic labels.
**Paper reference**: §3.2 "Semantic Gaussian Field", Eq. (1)-(8)

## Acceptance Criteria
- [ ] SG-Field parameter container matches paper attributes `(mu, Sigma, o, c, f)`
- [ ] Semantic feature rendering returns low-dimensional feature maps before decoding
- [ ] Decoder converts semantic features into per-class logits
- [ ] DSR regularizer is implemented as a standalone loss term
- [ ] Test: `uv run pytest tests/test_sg_field.py tests/test_decoder.py tests/test_regularization.py -v` passes

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_gs3lam/sg_field.py` | SG-Field state and initialization | Eq. (1)-(2) | ~220 |
| `src/anima_slam_gs3lam/rendering/rasterizer.py` | RGB/depth/semantic splatting wrapper | Eq. (3)-(7) | ~180 |
| `src/anima_slam_gs3lam/semantic/decoder.py` | 1x1 semantic decoder | Eq. (8) | ~90 |
| `src/anima_slam_gs3lam/losses/regularization.py` | DSR loss | Eq. (12) | ~80 |
| `tests/test_sg_field.py` | parameter and shape tests | — | ~120 |
| `tests/test_decoder.py` | decoder/logit tests | — | ~90 |
| `tests/test_regularization.py` | DSR behavior tests | — | ~90 |

## Architecture Detail (from paper)
### Inputs
- `means3d`: `Tensor[N,3]`
- `quat`: `Tensor[N,4]`
- `log_scales`: `Tensor[N,1]` for isotropic or `Tensor[N,3]` for anisotropic
- `opacity`: `Tensor[N,1]`
- `rgb`: `Tensor[N,3]`
- `semantic_feature`: `Tensor[N,16]`
- `camera_pose`: `Tensor[4,4]`
- `intrinsics`: `Tensor[3,3]`

### Outputs
- `render_rgb`: `Tensor[3,H,W]`
- `render_depth`: `Tensor[1,H,W]`
- `render_semantic_feature`: `Tensor[16,H,W]`
- `render_opacity`: `Tensor[1,H,W]`
- `semantic_logits`: `Tensor[256,H,W]`

### Algorithm
```python
# Paper §3.2, Eq. (1)-(8)
class SemanticGaussianField(nn.Module):
    def forward(self, pose: torch.Tensor, intrinsics: torch.Tensor, image_size: tuple[int, int]):
        return rasterize_semantic_gaussians(...)


class SemanticDecoder(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 256):
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

## Dependencies
```toml
torch = ">=2.0"
numpy = ">=1.26"
```

## Data Requirements
| Asset | Size | Path | Download |
|---|---|---|---|
| CUDA semantic rasterizer | source build | `repositories/GS3LAM/submodules/gaussian-semantic-rasterization/` | DONE |
| Decoder checkpoint (optional) | scene-dependent | `/mnt/forge-data/models/slam/gs3lam/decoder.pt` | MISSING |

## Test Plan
```bash
uv run pytest tests/test_sg_field.py tests/test_decoder.py tests/test_regularization.py -v
```

## References
- Paper: §3.2, Eq. (1)-(8), Fig. 3
- Reference impl: `repositories/GS3LAM/src/Mapper.py`, `repositories/GS3LAM/src/Decoder.py`, `repositories/GS3LAM/src/Render.py`
- Depends on: PRD-01
- Feeds into: PRD-03
