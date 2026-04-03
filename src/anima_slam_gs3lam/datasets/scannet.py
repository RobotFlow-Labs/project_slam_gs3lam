"""ScanNet semantic dataset adapter."""

from __future__ import annotations

from pathlib import Path

from anima_slam_gs3lam.types import FrameRecord

from .base import SemanticRGBDDataset


class ScannetSemanticDataset(SemanticRGBDDataset):
    def discover_records(self) -> list[FrameRecord]:
        scene_root = self.root / self.sequence
        color_paths = sorted((scene_root / "color").glob("*.jpg"))
        depth_paths = sorted((scene_root / "depth").glob("*.png"))
        semantic_paths = sorted((scene_root / "label-filt").glob("*.png"))
        pose_paths = sorted((scene_root / "pose").glob("*.txt"))

        if not (len(color_paths) == len(depth_paths) == len(semantic_paths) == len(pose_paths)):
            raise ValueError("ScanNet dataset is inconsistent across RGB, depth, semantic, and poses.")

        return [
            FrameRecord(
                frame_id=frame_id,
                sequence=self.sequence,
                rgb_path=Path(rgb_path),
                depth_path=Path(depth_path),
                semantic_path=Path(semantic_path),
                pose=self.load_pose_matrix(Path(pose_path)),
            )
            for frame_id, (rgb_path, depth_path, semantic_path, pose_path) in enumerate(
                zip(color_paths, depth_paths, semantic_paths, pose_paths, strict=True)
            )
        ]
