"""ROS2-friendly message conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tomllib
import torch

from anima_slam_gs3lam.api.schemas import StepFrameResponse, TensorPayload


@dataclass(frozen=True)
class TopicNames:
    rgb: str
    depth: str
    camera_info: str
    semantic: str
    pose: str
    semantic_map: str
    diagnostics: str


@dataclass(frozen=True)
class FrameNames:
    map: str
    camera: str


@dataclass(frozen=True)
class Ros2Config:
    session_id: str
    dataset_name: str
    sequence: str
    qos_depth: int
    topics: TopicNames
    frames: FrameNames


def load_ros2_config(path: str | Path = "configs/ros2.toml") -> Ros2Config:
    raw = tomllib.loads(Path(path).read_text())
    topics = raw["topics"]
    frames = raw["frames"]
    runtime = raw["runtime"]
    qos = raw["qos"]
    return Ros2Config(
        session_id=runtime["session_id"],
        dataset_name=runtime["dataset_name"],
        sequence=runtime["sequence"],
        qos_depth=qos["depth"],
        topics=TopicNames(
            rgb=topics["rgb"],
            depth=topics["depth"],
            camera_info=topics["camera_info"],
            semantic=topics["semantic"],
            pose=topics["pose"],
            semantic_map=topics["semantic_map"],
            diagnostics=topics["diagnostics"],
        ),
        frames=FrameNames(map=frames["map"], camera=frames["camera"]),
    )


def image_msg_to_tensor(msg: Any, *, normalize: bool = False) -> torch.Tensor:
    encoding = getattr(msg, "encoding", "")
    height = int(getattr(msg, "height"))
    width = int(getattr(msg, "width"))
    buffer = getattr(msg, "data")

    if encoding in {"rgb8", "bgr8"}:
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "bgr8":
            array = array[..., ::-1].copy()
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32)
        return tensor / 255.0 if normalize else tensor

    if encoding in {"mono8"}:
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
        tensor = torch.from_numpy(array.copy()).to(torch.uint8)
        return tensor.to(torch.float32) / 255.0 if normalize else tensor

    if encoding in {"16UC1", "mono16"}:
        array = np.frombuffer(buffer, dtype=np.uint16).reshape(height, width)
        return torch.from_numpy(array.copy()).to(torch.float32).unsqueeze(0)

    if encoding == "32FC1":
        array = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)
        return torch.from_numpy(array.copy()).to(torch.float32).unsqueeze(0)

    raise ValueError(f"Unsupported image encoding: {encoding}")


def camera_info_to_matrix(msg: Any) -> torch.Tensor:
    values = list(getattr(msg, "k"))
    if len(values) != 9:
        raise ValueError("camera_info.k must contain 9 values")
    return torch.tensor(values, dtype=torch.float32).reshape(3, 3)


def tensor_to_image_message(
    tensor_payload: TensorPayload,
    *,
    encoding: str,
    frame_id: str,
) -> dict[str, Any]:
    return {
        "encoding": encoding,
        "frame_id": frame_id,
        "tensor": tensor_payload.model_dump(),
    }


def pose_matrix_to_pose_message(pose: TensorPayload, *, frame_id: str) -> dict[str, Any]:
    matrix = pose.to_tensor(dtype=torch.float32)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    quaternion = _rotation_matrix_to_quaternion(rotation)
    return {
        "frame_id": frame_id,
        "position": {
            "x": float(translation[0].item()),
            "y": float(translation[1].item()),
            "z": float(translation[2].item()),
        },
        "orientation": {
            "x": float(quaternion[1].item()),
            "y": float(quaternion[2].item()),
            "z": float(quaternion[3].item()),
            "w": float(quaternion[0].item()),
        },
    }


def diagnostics_message(response: StepFrameResponse, *, frame_id: str) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "bootstrapped": response.bootstrapped,
        "new_gaussians": response.new_gaussians,
        "tracking_loss": response.tracking_loss,
        "mapping_loss": response.mapping_loss,
        "timings_ms": response.timings_ms.model_dump(),
    }


def _rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.trace()
    if trace > 0.0:
        scale = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
        return torch.tensor([qw, qx, qy, qz], dtype=rotation.dtype)

    diagonal = rotation.diagonal()
    max_index = int(torch.argmax(diagonal).item())
    if max_index == 0:
        scale = torch.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return torch.tensor(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ],
            dtype=rotation.dtype,
        )
    if max_index == 1:
        scale = torch.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return torch.tensor(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ],
            dtype=rotation.dtype,
        )
    scale = torch.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return torch.tensor(
        [
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        ],
        dtype=rotation.dtype,
    )
