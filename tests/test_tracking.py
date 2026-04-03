import torch

from anima_slam_gs3lam.tracking.tracker import compute_observed_mask, initialize_pose


def test_constant_velocity_pose_init():
    prev_prev = torch.eye(4)
    prev = torch.eye(4)
    prev[0, 3] = 1.0
    pose = initialize_pose(prev, prev_prev)
    assert torch.isclose(pose[0, 3], torch.tensor(2.0))


def test_observed_mask_requires_high_opacity_and_small_error():
    opacity = torch.tensor([[1.0, 0.2], [1.0, 1.0]])
    depth_render = torch.tensor([[1.0, 1.0], [1.0, 5.0]])
    depth_gt = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    mask = compute_observed_mask(opacity, depth_render, depth_gt, tau_obs=0.9)
    assert mask[0, 0]
    assert not mask[0, 1]
    assert not mask[1, 1]
