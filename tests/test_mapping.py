import torch

from anima_slam_gs3lam.losses.mapping import mapping_loss
from anima_slam_gs3lam.mapping.expansion import compute_unobserved_mask, expand_field_from_frame
from anima_slam_gs3lam.rendering.rasterizer import RenderOutputs
from anima_slam_gs3lam.sg_field import SGFieldInit, SemanticGaussianField
from anima_slam_gs3lam.types import FrameBatch


def _frame() -> FrameBatch:
    rgb = torch.ones(3, 8, 8, dtype=torch.float32) * 0.5
    depth = torch.ones(1, 8, 8, dtype=torch.float32)
    semantic = torch.zeros(8, 8, dtype=torch.int64)
    intrinsics = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32)
    return FrameBatch(rgb=rgb, depth=depth, semantic=semantic, intrinsics=intrinsics, pose=pose, frame_id=0, sequence="scene")


def test_unobserved_mask_matches_alpha_or_depth_violation():
    opacity = torch.tensor([[0.0, 0.5], [1.0, 1.0]])
    depth_render = torch.tensor([[1.0, 2.0], [1.0, 5.0]])
    depth_gt = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    mask = compute_unobserved_mask(opacity, depth_render, depth_gt, tau_unobs=0.1)
    assert mask[0, 0]
    assert mask[1, 1]


def test_expand_field_from_frame_adds_gaussians():
    frame = _frame()
    field = SemanticGaussianField(SGFieldInit(num_gaussians=1, semantic_dim=16))
    with torch.no_grad():
        field.means3d[0] = torch.tensor([0.0, 0.0, 1.0])
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:4, 2:4] = True
    added = expand_field_from_frame(field, frame, mask)
    assert added == 4
    assert field.num_gaussians == 5


def test_mapping_loss_combines_render_and_regularization_terms():
    render = RenderOutputs(
        rgb=torch.zeros(3, 8, 8),
        depth=torch.ones(1, 8, 8),
        semantic_feature=torch.zeros(16, 8, 8),
        opacity=torch.ones(1, 8, 8),
    )
    logits = torch.zeros(256, 8, 8)
    total, terms = mapping_loss(
        render,
        target_rgb=torch.ones(3, 8, 8),
        target_depth=torch.ones(1, 8, 8),
        semantic_logits=logits,
        target_semantic=torch.zeros(8, 8, dtype=torch.long),
        scales=torch.tensor([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]]),
    )
    assert total > 0
    assert set(terms) == {"color", "depth", "semantic", "big_scale", "small_scale"}
