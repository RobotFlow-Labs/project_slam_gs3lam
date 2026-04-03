"""Replica semantic dataset adapter."""

from __future__ import annotations

from pathlib import Path

import torch

from anima_slam_gs3lam.types import FrameRecord

from .base import SemanticRGBDDataset


class ReplicaSemanticDataset(SemanticRGBDDataset):
    def discover_records(self) -> list[FrameRecord]:
        scene_root = self.root / self.sequence
        color_paths = sorted((scene_root / "results").glob("frame*.jpg"))
        depth_paths = sorted((scene_root / "results").glob("depth*.png"))
        # Only match semantic_class_N.png, not vis_sem_class_N.png
        semantic_paths = sorted(
            p for p in (scene_root / "semantic_class").glob("semantic_class_*.png")
            if not p.name.startswith("vis_")
        )
        poses = self._load_poses(scene_root / "traj.txt", len(color_paths))

        n_frames = min(len(color_paths), len(depth_paths), len(semantic_paths), len(poses))
        if n_frames == 0:
            raise ValueError(
                f"Replica scene {self.sequence} has no matching frames. "
                f"rgb={len(color_paths)} depth={len(depth_paths)} sem={len(semantic_paths)} poses={len(poses)}"
            )

        records: list[FrameRecord] = []
        for frame_id in range(n_frames):
            records.append(
                FrameRecord(
                    frame_id=frame_id,
                    sequence=self.sequence,
                    rgb_path=color_paths[frame_id],
                    depth_path=depth_paths[frame_id],
                    semantic_path=semantic_paths[frame_id],
                    pose=poses[frame_id],
                )
            )
        return records

    @staticmethod
    def _load_poses(traj_path: Path, expected_count: int) -> list[torch.Tensor]:
        """Load poses from traj.txt, supporting both 4x4 matrix and TUM quaternion formats."""
        lines = traj_path.read_text().strip().splitlines()
        if not lines:
            return []

        tokens_first = lines[0].split()
        if len(tokens_first) == 16:
            # 4x4 matrix format: 16 floats per line
            return [
                torch.tensor([float(t) for t in line.split()], dtype=torch.float32).reshape(4, 4)
                for line in lines
            ]
        elif len(tokens_first) in (7, 8):
            # TUM format: [timestamp] tx ty tz qx qy qz qw
            poses = []
            for line in lines:
                vals = [float(t) for t in line.split()]
                if len(vals) == 8:
                    _, tx, ty, tz, qx, qy, qz, qw = vals
                else:
                    tx, ty, tz, qx, qy, qz, qw = vals
                poses.append(_quat_to_pose(tx, ty, tz, qx, qy, qz, qw))

            # Interpolate if fewer poses than frames
            if len(poses) < expected_count and len(poses) > 1:
                poses = _interpolate_poses(poses, expected_count)
            return poses
        else:
            raise ValueError(
                f"Unknown traj.txt format: {len(tokens_first)} tokens per line (expected 7, 8, or 16)"
            )


def _quat_to_pose(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> torch.Tensor:
    """Convert quaternion + translation to 4x4 c2w matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 0] = 1 - 2 * (yy + zz)
    pose[0, 1] = 2 * (xy - wz)
    pose[0, 2] = 2 * (xz + wy)
    pose[1, 0] = 2 * (xy + wz)
    pose[1, 1] = 1 - 2 * (xx + zz)
    pose[1, 2] = 2 * (yz - wx)
    pose[2, 0] = 2 * (xz - wy)
    pose[2, 1] = 2 * (yz + wx)
    pose[2, 2] = 1 - 2 * (xx + yy)
    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


def _interpolate_poses(poses: list[torch.Tensor], target_count: int) -> list[torch.Tensor]:
    """Linearly interpolate poses to fill gaps (simple translation + nearest rotation)."""
    n = len(poses)
    result = []
    for i in range(target_count):
        frac = i * (n - 1) / (target_count - 1)
        lo = min(int(frac), n - 2)
        hi = lo + 1
        alpha = frac - lo
        # Interpolate translation, use nearest rotation
        interp = poses[lo].clone()
        interp[:3, 3] = (1 - alpha) * poses[lo][:3, 3] + alpha * poses[hi][:3, 3]
        if alpha > 0.5:
            interp[:3, :3] = poses[hi][:3, :3]
        result.append(interp)
    return result
