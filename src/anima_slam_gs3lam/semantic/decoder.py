"""Semantic feature decoder for GS3LAM."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SemanticDecoder(nn.Module):
    """Paper-faithful 1x1 semantic projection from low-dimensional features to class logits."""

    def __init__(self, in_channels: int = 16, out_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 3:
            raise ValueError("feature_map must have shape [C, H, W]")
        if feature_map.shape[0] != self.in_channels:
            raise ValueError(
                f"feature_map channel dimension {feature_map.shape[0]} does not match decoder input {self.in_channels}"
            )
        return self.proj(feature_map.unsqueeze(0)).squeeze(0)

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.load_state_dict(checkpoint, strict=strict)
