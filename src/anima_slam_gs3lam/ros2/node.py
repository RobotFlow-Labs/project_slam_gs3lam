"""ROS2 runtime wrapper with a local shim for non-ROS test environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from anima_slam_gs3lam.api.schemas import LoadSceneRequest, StepFrameRequest, TensorPayload
from anima_slam_gs3lam.api.service import SessionService
from anima_slam_gs3lam.ros2.messages import (
    camera_info_to_matrix,
    diagnostics_message,
    image_msg_to_tensor,
    load_ros2_config,
    pose_matrix_to_pose_message,
    tensor_to_image_message,
)

try:  # pragma: no cover - exercised on a ROS2 host
    from rclpy.node import Node as _RosNode
except Exception:  # pragma: no cover - local Mac test shim
    class _LoggerShim:
        def info(self, _message: str) -> None:
            return None

        def warning(self, _message: str) -> None:
            return None

        def error(self, _message: str) -> None:
            return None

    @dataclass
    class _PublisherShim:
        topic: str
        published: list[Any] = field(default_factory=list)

        def publish(self, msg: Any) -> None:
            self.published.append(msg)

    class _RosNode:
        def __init__(self, name: str) -> None:
            self.node_name = name
            self._logger = _LoggerShim()

        def create_subscription(self, _msg_type: Any, topic: str, callback: Any, _qos_depth: int) -> dict[str, Any]:
            return {"topic": topic, "callback": callback}

        def create_publisher(self, _msg_type: Any, topic: str, _qos_depth: int) -> _PublisherShim:
            return _PublisherShim(topic=topic)

        def get_logger(self) -> _LoggerShim:
            return self._logger


class GS3LAMNode(_RosNode):
    """ROS2-facing node that wraps the API session service."""

    def __init__(
        self,
        *,
        service: SessionService | None = None,
        config_path: str | Path = "configs/ros2.toml",
    ) -> None:
        super().__init__("gs3lam")
        self.service = service or SessionService()
        self.config = load_ros2_config(config_path)
        self.service.load_scene(
            LoadSceneRequest(
                session_id=self.config.session_id,
                dataset_name=self.config.dataset_name,  # type: ignore[arg-type]
                sequence_override=self.config.sequence,
            )
        )

        self.rgb_subscription = self.create_subscription(
            object, self.config.topics.rgb, self._unsupported_async_callback, self.config.qos_depth
        )
        self.depth_subscription = self.create_subscription(
            object, self.config.topics.depth, self._unsupported_async_callback, self.config.qos_depth
        )
        self.semantic_subscription = self.create_subscription(
            object, self.config.topics.semantic, self._unsupported_async_callback, self.config.qos_depth
        )
        self.camera_info_subscription = self.create_subscription(
            object, self.config.topics.camera_info, self._unsupported_async_callback, self.config.qos_depth
        )

        self.pose_publisher = self.create_publisher(object, self.config.topics.pose, self.config.qos_depth)
        self.semantic_map_publisher = self.create_publisher(
            object, self.config.topics.semantic_map, self.config.qos_depth
        )
        self.diagnostics_publisher = self.create_publisher(
            object, self.config.topics.diagnostics, self.config.qos_depth
        )

    def on_frame(
        self,
        rgb_msg: Any,
        depth_msg: Any,
        semantic_msg: Any,
        camera_info_msg: Any,
        *,
        frame_id: int,
    ):
        request = StepFrameRequest(
            session_id=self.config.session_id,
            frame_id=frame_id,
            sequence=self.config.sequence,
            rgb=TensorPayload.from_tensor(image_msg_to_tensor(rgb_msg, normalize=True)),
            depth=TensorPayload.from_tensor(image_msg_to_tensor(depth_msg)),
            semantic=TensorPayload.from_tensor(image_msg_to_tensor(semantic_msg).to(dtype=torch.int64)),
            intrinsics=TensorPayload.from_tensor(camera_info_to_matrix(camera_info_msg)),
            pose=TensorPayload.from_tensor(_identity_pose()),
        )
        response = self.service.step(request)

        self.pose_publisher.publish(
            pose_matrix_to_pose_message(response.pose_w2c, frame_id=self.config.frames.map)
        )
        self.semantic_map_publisher.publish(
            tensor_to_image_message(
                response.semantic_logits,
                encoding="32FC1",
                frame_id=self.config.frames.map,
            )
        )
        self.diagnostics_publisher.publish(diagnostics_message(response, frame_id=str(frame_id)))
        return response

    def _unsupported_async_callback(self, *_args: Any, **_kwargs: Any) -> None:
        self.get_logger().info("Use GS3LAMNode.on_frame() with synchronized RGB-D-semantic inputs.")


def _identity_pose():
    import torch

    return torch.eye(4, dtype=torch.float32)
