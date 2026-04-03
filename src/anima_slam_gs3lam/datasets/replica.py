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
        semantic_paths = sorted((scene_root / "semantic_class").glob("semantic_class_*.png"))
        pose_lines = (scene_root / "traj.txt").read_text().strip().splitlines()

        if not (len(color_paths) == len(depth_paths) == len(semantic_paths) == len(pose_lines)):
            raise ValueError("Replica dataset is inconsistent across RGB, depth, semantic, and poses.")

        records: list[FrameRecord] = []
        for frame_id, (rgb_path, depth_path, semantic_path, pose_line) in enumerate(
            zip(color_paths, depth_paths, semantic_paths, pose_lines, strict=True)
        ):
            pose = torch.tensor([float(token) for token in pose_line.split()], dtype=torch.float32).reshape(4, 4)
            records.append(
                FrameRecord(
                    frame_id=frame_id,
                    sequence=self.sequence,
                    rgb_path=Path(rgb_path),
                    depth_path=Path(depth_path),
                    semantic_path=Path(semantic_path),
                    pose=pose,
                )
            )
        return records
