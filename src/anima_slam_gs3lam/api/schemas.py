"""Typed API contracts for GS3LAM service mode."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field


class TensorPayload(BaseModel):
    """Serializable tensor wrapper used by the FastAPI surface."""

    model_config = ConfigDict(frozen=True)

    shape: list[int]
    data: list[float]
    dtype: str = "float32"

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorPayload":
        cpu_tensor = tensor.detach().cpu()
        return cls(
            shape=list(cpu_tensor.shape),
            data=cpu_tensor.reshape(-1).tolist(),
            dtype=str(cpu_tensor.dtype).replace("torch.", ""),
        )

    def to_tensor(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor_dtype = dtype or getattr(torch, self.dtype)
        tensor = torch.tensor(self.data, dtype=tensor_dtype)
        return tensor.reshape(self.shape)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "anima-slam-gs3lam"


class LoadSceneRequest(BaseModel):
    session_id: str = "default"
    dataset_name: Literal["replica", "scannet", "tum"] = "replica"
    use_repo_overrides: bool = False
    root_override: str | None = None
    sequence_override: str | None = None


class LoadSceneResponse(BaseModel):
    session_id: str
    dataset_name: str
    sequence: str
    semantic_dim: int
    semantic_classes: int
    backend: str


class StepFrameRequest(BaseModel):
    session_id: str = "default"
    frame_id: int
    sequence: str = "runtime"
    rgb: TensorPayload
    depth: TensorPayload
    semantic: TensorPayload
    intrinsics: TensorPayload
    pose: TensorPayload


class TimingBreakdown(BaseModel):
    step_ms: float
    render_ms: float
    total_ms: float


class StepFrameResponse(BaseModel):
    session_id: str
    frame_id: int
    bootstrapped: bool
    pose_w2c: TensorPayload
    render_rgb: TensorPayload
    render_depth: TensorPayload
    semantic_logits: TensorPayload
    new_gaussians: int
    tracking_loss: float | None = None
    mapping_loss: float | None = None
    timings_ms: TimingBreakdown


class SnapshotResponse(BaseModel):
    session_id: str
    dataset_name: str
    sequence: str
    frames_processed: int
    num_gaussians: int
    num_keyframes: int
    num_poses: int
    last_pose_w2c: TensorPayload | None = None
    checkpoint_paths: list[str] = Field(default_factory=list)
