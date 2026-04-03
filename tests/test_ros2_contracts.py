from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from anima_slam_gs3lam.ros2.messages import (
    camera_info_to_matrix,
    image_msg_to_tensor,
    load_ros2_config,
)
from anima_slam_gs3lam.ros2.node import GS3LAMNode


@dataclass
class StubImage:
    height: int
    width: int
    encoding: str
    data: bytes


@dataclass
class StubCameraInfo:
    k: list[float]


def test_topic_config_loaded() -> None:
    config = load_ros2_config()
    assert config.topics.rgb == "/camera/rgb/image"
    assert config.topics.pose == "/slam/pose"
    assert config.session_id == "ros2"


def test_image_and_camera_conversions() -> None:
    rgb_array = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    rgb_msg = StubImage(height=2, width=2, encoding="rgb8", data=rgb_array.tobytes())
    rgb_tensor = image_msg_to_tensor(rgb_msg, normalize=True)
    assert rgb_tensor.shape == (3, 2, 2)
    assert torch.isclose(rgb_tensor[0, 0, 0], torch.tensor(1.0))

    camera_info = StubCameraInfo(k=[2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0])
    intrinsics = camera_info_to_matrix(camera_info)
    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 0].item() == 2.0


def test_ros2_node_publishes_contracts() -> None:
    node = GS3LAMNode()

    rgb = np.ones((2, 2, 3), dtype=np.uint8) * 255
    depth = np.ones((2, 2), dtype=np.float32)
    semantic = np.zeros((2, 2), dtype=np.uint8)
    camera_info = StubCameraInfo(k=[2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0])

    response = node.on_frame(
        StubImage(height=2, width=2, encoding="rgb8", data=rgb.tobytes()),
        StubImage(height=2, width=2, encoding="32FC1", data=depth.tobytes()),
        StubImage(height=2, width=2, encoding="mono8", data=semantic.tobytes()),
        camera_info,
        frame_id=0,
    )

    assert response.frame_id == 0
    assert len(node.pose_publisher.published) == 1
    assert len(node.semantic_map_publisher.published) == 1
    assert len(node.diagnostics_publisher.published) == 1
