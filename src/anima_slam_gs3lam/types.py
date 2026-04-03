"""Shared typed contracts for GS3LAM data flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    png_depth_scale: float
    crop_edge: int = 0

    def as_matrix(self) -> torch.Tensor:
        return torch.tensor(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    sequence: str
    rgb_path: Path
    depth_path: Path
    semantic_path: Path
    pose: torch.Tensor


class FrameBatch(NamedTuple):
    rgb: torch.Tensor
    depth: torch.Tensor
    semantic: torch.Tensor
    intrinsics: torch.Tensor
    pose: torch.Tensor
    frame_id: int
    sequence: str

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[0] != 3:
            raise ValueError("rgb must have shape [3, H, W]")
        if self.depth.ndim != 3 or self.depth.shape[0] != 1:
            raise ValueError("depth must have shape [1, H, W]")
        if self.semantic.ndim != 2:
            raise ValueError("semantic must have shape [H, W]")
        if self.pose.shape != (4, 4):
            raise ValueError("pose must have shape [4, 4]")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape [3, 3]")
