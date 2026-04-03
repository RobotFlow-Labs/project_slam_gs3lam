from pathlib import Path

import torch

from anima_slam_gs3lam.mapping.rskm import sample_keyframes
from anima_slam_gs3lam.pipeline.slam_loop import GS3LAMLoop
from anima_slam_gs3lam.types import FrameBatch


def _frame(frame_id: int) -> FrameBatch:
    rgb = torch.zeros(3, 8, 8, dtype=torch.float32)
    depth = torch.ones(1, 8, 8, dtype=torch.float32)
    semantic = torch.zeros(8, 8, dtype=torch.int64)
    intrinsics = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 3] = frame_id * 0.05
    return FrameBatch(rgb=rgb, depth=depth, semantic=semantic, intrinsics=intrinsics, pose=pose, frame_id=frame_id, sequence="synthetic")


def test_rskm_sampling_is_deterministic():
    keyframes = list(range(10))
    sample_a = sample_keyframes(keyframes, step_idx=5, t_opt=4, seed=7)
    sample_b = sample_keyframes(keyframes, step_idx=5, t_opt=4, seed=7)
    assert sample_a == sample_b
    assert sample_a[-1] == keyframes[-1]


def test_slam_loop_writes_checkpoint(tmp_path: Path):
    loop = GS3LAMLoop()
    loop.step(_frame(0))
    loop.step(_frame(1))
    checkpoint = loop.save_checkpoint(tmp_path, step_idx=1)
    assert checkpoint.exists()
