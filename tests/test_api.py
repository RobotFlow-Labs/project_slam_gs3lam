from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from anima_slam_gs3lam.api.app import app
from anima_slam_gs3lam.api.schemas import TensorPayload


def _tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    return TensorPayload.from_tensor(tensor).model_dump()


def test_schema_roundtrip() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    payload = TensorPayload.from_tensor(tensor)
    restored = payload.to_tensor(dtype=torch.float32)
    assert torch.equal(restored, tensor)


def test_api_session_lifecycle() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    load_response = client.post(
        "/v1/gs3lam/load-scene",
        json={"session_id": "demo", "dataset_name": "replica"},
    )
    assert load_response.status_code == 200
    assert load_response.json()["session_id"] == "demo"

    rgb = torch.ones(3, 2, 2, dtype=torch.float32)
    depth = torch.full((1, 2, 2), 1.0, dtype=torch.float32)
    semantic = torch.zeros(2, 2, dtype=torch.int64)
    intrinsics = torch.tensor(
        [
            [2.0, 0.0, 0.5],
            [0.0, 2.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pose = torch.eye(4, dtype=torch.float32)

    step_response = client.post(
        "/v1/gs3lam/step-frame",
        json={
            "session_id": "demo",
            "frame_id": 0,
            "sequence": "office0",
            "rgb": _tensor_payload(rgb),
            "depth": _tensor_payload(depth),
            "semantic": _tensor_payload(semantic),
            "intrinsics": _tensor_payload(intrinsics),
            "pose": _tensor_payload(pose),
        },
    )
    assert step_response.status_code == 200
    body = step_response.json()
    assert body["pose_w2c"]["shape"] == [4, 4]
    assert body["render_rgb"]["shape"] == [3, 2, 2]
    assert body["render_depth"]["shape"] == [1, 2, 2]
    assert body["semantic_logits"]["shape"] == [256, 2, 2]
    assert body["timings_ms"]["total_ms"] >= 0.0

    snapshot = client.get("/v1/gs3lam/snapshot/demo")
    assert snapshot.status_code == 200
    snapshot_body = snapshot.json()
    assert snapshot_body["frames_processed"] == 1
    assert snapshot_body["num_gaussians"] > 0
    assert snapshot_body["num_keyframes"] == 1
    assert snapshot_body["num_poses"] == 1
