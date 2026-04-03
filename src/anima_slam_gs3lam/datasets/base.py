"""Shared RGB-D-semantic dataset utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from anima_slam_gs3lam.config import DatasetPreset
from anima_slam_gs3lam.types import CameraIntrinsics, FrameBatch, FrameRecord


class SemanticRGBDDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetPreset) -> None:
        self.config = config
        self.root = Path(config.root)
        self.sequence = config.sequence
        camera = config.camera
        self.camera = CameraIntrinsics(
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            width=config.desired_image_width,
            height=config.desired_image_height,
            png_depth_scale=camera.png_depth_scale,
            crop_edge=camera.crop_edge,
        )
        self.records = self._slice_records(self.discover_records())

    def _slice_records(self, records: Sequence[FrameRecord]) -> list[FrameRecord]:
        start = self.config.start
        end = None if self.config.end == -1 else self.config.end
        stride = self.config.stride
        return list(records[start:end:stride])

    def discover_records(self) -> list[FrameRecord]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FrameBatch:
        record = self.records[index]
        batch = FrameBatch(
            rgb=self._load_rgb(record.rgb_path),
            depth=self._load_depth(record.depth_path),
            semantic=self._load_semantic(record.semantic_path),
            intrinsics=self.camera.as_matrix(),
            pose=record.pose.clone().to(dtype=torch.float32),
            frame_id=record.frame_id,
            sequence=record.sequence,
        )
        batch.validate()
        return batch

    def _resize(self, image: Image.Image, *, mode: str) -> Image.Image:
        resample = Image.Resampling.BILINEAR if mode == "rgb" else Image.Resampling.NEAREST
        return image.resize(
            (self.config.desired_image_width, self.config.desired_image_height),
            resample=resample,
        )

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = self._resize(image, mode="rgb")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_depth(self, path: Path) -> torch.Tensor:
        image = Image.open(path)
        image = self._resize(image, mode="depth")
        array = np.asarray(image, dtype=np.float32) / self.camera.png_depth_scale
        return torch.from_numpy(array).unsqueeze(0)

    def _load_semantic(self, path: Path) -> torch.Tensor:
        image = Image.open(path)
        image = self._resize(image, mode="semantic")
        array = np.asarray(image, dtype=np.int64)
        return torch.from_numpy(array)

    @staticmethod
    def load_pose_matrix(path: Path) -> torch.Tensor:
        return torch.from_numpy(np.loadtxt(path, dtype=np.float32))
