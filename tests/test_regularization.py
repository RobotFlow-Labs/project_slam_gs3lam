import torch

from anima_slam_gs3lam.losses.regularization import depth_adaptive_scale_regularization


def test_dsr_is_zero_for_uniform_scales():
    scales = torch.full((5, 3), 0.1)
    penalties = depth_adaptive_scale_regularization(scales)
    assert torch.isclose(penalties["loss"], torch.tensor(0.0))


def test_dsr_penalizes_outliers():
    scales = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.11, 0.11, 0.11],
            [0.09, 0.09, 0.09],
            [2.0, 2.0, 2.0],
            [0.0001, 0.0001, 0.0001],
        ],
        dtype=torch.float32,
    )
    penalties = depth_adaptive_scale_regularization(scales, num_std=1.0)
    assert penalties["big_gaussian_reg"] > 0
    assert penalties["small_gaussian_reg"] > 0
    assert penalties["loss"] > 0
