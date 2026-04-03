import torch

from anima_slam_gs3lam.sg_field import SGFieldInit, SemanticGaussianField


def test_sg_field_shapes():
    field = SemanticGaussianField(SGFieldInit(num_gaussians=8, semantic_dim=16))
    assert field.means3d.shape == (8, 3)
    assert field.quaternions.shape == (8, 4)
    assert field.log_scales.shape == (8, 1)
    assert field.semantic_features.shape == (8, 16)


def test_sg_field_from_point_cloud_initializes_state():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.2, 2.0]], dtype=torch.float32)
    rgb = torch.tensor([[1.0, 0.5, 0.25], [0.2, 0.3, 0.4]], dtype=torch.float32)
    mean_sq_dist = torch.tensor([0.01, 0.04], dtype=torch.float32)

    field = SemanticGaussianField.from_point_cloud(xyz, rgb, mean_sq_dist=mean_sq_dist)
    assert field.num_gaussians == 2
    assert torch.allclose(field.means3d, xyz)
    assert torch.allclose(field.rgb, rgb)
    assert torch.allclose(field.scales()[:, 0], torch.sqrt(mean_sq_dist), atol=1e-6)


def test_sg_field_forward_returns_render_outputs():
    field = SemanticGaussianField(SGFieldInit(num_gaussians=4, semantic_dim=16))
    with torch.no_grad():
        field.means3d[:, 2] = 1.0
        field.rgb[:, 0] = 1.0
    pose = torch.eye(4)
    intrinsics = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    outputs = field(pose=pose, intrinsics=intrinsics, image_size=(8, 8))
    assert outputs.rgb.shape == (3, 8, 8)
    assert outputs.depth.shape == (1, 8, 8)
    assert outputs.semantic_feature.shape == (16, 8, 8)
    assert outputs.opacity.shape == (1, 8, 8)
