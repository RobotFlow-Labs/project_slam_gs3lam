"""TUM-RGBD semantic dataset adapter with pseudo semantic masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from anima_slam_gs3lam.types import FrameRecord

from .base import SemanticRGBDDataset


class TUMSemanticDataset(SemanticRGBDDataset):
    def discover_records(self) -> list[FrameRecord]:
        scene_root = self.root / self.sequence
        pose_list = scene_root / "groundtruth.txt"
        if not pose_list.exists():
            pose_list = scene_root / "pose.txt"

        image_data = self._parse_list(scene_root / "rgb.txt")
        depth_data = self._parse_list(scene_root / "depth.txt")
        pose_data = self._parse_list(pose_list, skiprows=1)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        associations = self._associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
        sampled_indices = self._sample_associations(associations, tstamp_image)

        records: list[FrameRecord] = []
        for frame_id, assoc_index in enumerate(sampled_indices):
            image_idx, depth_idx, pose_idx = associations[assoc_index]
            rgb_rel = image_data[image_idx, 1]
            rgb_path = scene_root / rgb_rel
            depth_path = scene_root / depth_data[depth_idx, 1]
            semantic_path = scene_root / rgb_rel.replace("rgb", "object_mask")
            records.append(
                FrameRecord(
                    frame_id=frame_id,
                    sequence=self.sequence,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    semantic_path=semantic_path,
                    pose=self._pose_matrix_from_quaternion(pose_vecs[pose_idx]),
                )
            )
        return records

    @staticmethod
    def _parse_list(path: Path, skiprows: int = 0) -> np.ndarray:
        return np.atleast_2d(np.loadtxt(path, delimiter=" ", dtype=np.str_, skiprows=skiprows))

    @staticmethod
    def _associate_frames(
        tstamp_image: np.ndarray,
        tstamp_depth: np.ndarray,
        tstamp_pose: np.ndarray,
        max_dt: float = 0.08,
    ) -> list[tuple[int, int, int]]:
        associations: list[tuple[int, int, int]] = []
        for image_idx, timestamp in enumerate(tstamp_image):
            depth_idx = int(np.argmin(np.abs(tstamp_depth - timestamp)))
            pose_idx = int(np.argmin(np.abs(tstamp_pose - timestamp)))
            if (
                abs(tstamp_depth[depth_idx] - timestamp) < max_dt
                and abs(tstamp_pose[pose_idx] - timestamp) < max_dt
            ):
                associations.append((image_idx, depth_idx, pose_idx))
        return associations

    @staticmethod
    def _sample_associations(
        associations: list[tuple[int, int, int]],
        tstamp_image: np.ndarray,
        frame_rate: int = 32,
    ) -> list[int]:
        if not associations:
            return []
        sampled = [0]
        for index in range(1, len(associations)):
            prev_timestamp = tstamp_image[associations[sampled[-1]][0]]
            current_timestamp = tstamp_image[associations[index][0]]
            if current_timestamp - prev_timestamp > 1.0 / frame_rate:
                sampled.append(index)
        return sampled

    @staticmethod
    def _pose_matrix_from_quaternion(pvec: np.ndarray) -> torch.Tensor:
        x, y, z, qx, qy, qz, qw = pvec
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz

        rotation = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation
        pose[:3, 3] = np.array([x, y, z], dtype=np.float32)
        return torch.from_numpy(pose)
