"""In-memory GS3LAM service wrapper for FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from anima_slam_gs3lam.api.schemas import (
    LoadSceneRequest,
    LoadSceneResponse,
    SnapshotResponse,
    StepFrameRequest,
    StepFrameResponse,
    TensorPayload,
    TimingBreakdown,
)
from anima_slam_gs3lam.config import GS3LAMConfig, RuntimeConfig, default_config
from anima_slam_gs3lam.device import BackendInfo, detect_backend
from anima_slam_gs3lam.pipeline.slam_loop import GS3LAMLoop
from anima_slam_gs3lam.types import FrameBatch


@dataclass
class SessionState:
    session_id: str
    runtime: RuntimeConfig
    loop: GS3LAMLoop
    frames_processed: int = 0


class SessionService:
    """Owns process-local GS3LAM sessions and adapts them to API responses."""

    def __init__(self, config: GS3LAMConfig | None = None) -> None:
        self.config = config or default_config()
        self.backend = self._resolve_backend(self.config.compute.backend)
        self.sessions: dict[str, SessionState] = {}

    def load_scene(self, request: LoadSceneRequest) -> LoadSceneResponse:
        runtime = self.config.build_runtime_config(
            request.dataset_name,
            use_repo_overrides=request.use_repo_overrides,
            root_override=request.root_override,
            sequence_override=request.sequence_override,
        )
        loop = GS3LAMLoop(
            semantic_dim=self.config.paper_defaults.semantic.feature_dim,
            semantic_classes=self.config.paper_defaults.semantic.class_dim,
            keyframe_window=runtime.mapping.mapping_window_size,
        )
        self.sessions[request.session_id] = SessionState(
            session_id=request.session_id,
            runtime=runtime,
            loop=loop,
        )
        return LoadSceneResponse(
            session_id=request.session_id,
            dataset_name=request.dataset_name,
            sequence=runtime.dataset.sequence,
            semantic_dim=loop.semantic_dim,
            semantic_classes=loop.semantic_classes,
            backend=self.backend.accelerator,
        )

    def step(self, request: StepFrameRequest) -> StepFrameResponse:
        session = self.sessions.get(request.session_id)
        if session is None:
            self.load_scene(LoadSceneRequest(session_id=request.session_id))
            session = self.sessions[request.session_id]

        frame = self._to_frame_batch(request)

        step_start = perf_counter()
        result = session.loop.step(frame)
        step_end = perf_counter()

        render_start = perf_counter()
        pose = result["pose"]
        render = session.loop.state.field(
            pose=pose,
            intrinsics=frame.intrinsics,
            image_size=tuple(frame.depth.shape[-2:]),
        )
        semantic_logits = session.loop.state.decoder(render.semantic_feature)
        render_end = perf_counter()

        session.frames_processed += 1

        return StepFrameResponse(
            session_id=request.session_id,
            frame_id=request.frame_id,
            bootstrapped=bool(result["bootstrapped"]),
            pose_w2c=TensorPayload.from_tensor(pose),
            render_rgb=TensorPayload.from_tensor(render.rgb),
            render_depth=TensorPayload.from_tensor(render.depth),
            semantic_logits=TensorPayload.from_tensor(semantic_logits),
            new_gaussians=int(result["new_gaussians"]),
            tracking_loss=self._maybe_float(result.get("tracking_loss")),
            mapping_loss=self._maybe_float(result.get("mapping_loss")),
            timings_ms=TimingBreakdown(
                step_ms=(step_end - step_start) * 1000.0,
                render_ms=(render_end - render_start) * 1000.0,
                total_ms=(render_end - step_start) * 1000.0,
            ),
        )

    def snapshot(self, session_id: str) -> SnapshotResponse:
        session = self._require_session(session_id)
        last_pose = session.loop.state.poses[-1] if session.loop.state.poses else None
        field = session.loop.state.field
        return SnapshotResponse(
            session_id=session_id,
            dataset_name=session.runtime.dataset.name,
            sequence=session.runtime.dataset.sequence,
            frames_processed=session.frames_processed,
            num_gaussians=0 if field is None else field.num_gaussians,
            num_keyframes=len(session.loop.state.keyframes),
            num_poses=len(session.loop.state.poses),
            last_pose_w2c=None if last_pose is None else TensorPayload.from_tensor(last_pose),
            checkpoint_paths=[str(path) for path in session.loop.state.checkpoints],
        )

    def _require_session(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id={session_id!r}")
        return session

    def _to_frame_batch(self, request: StepFrameRequest) -> FrameBatch:
        frame = FrameBatch(
            rgb=request.rgb.to_tensor(dtype=torch.float32),
            depth=request.depth.to_tensor(dtype=torch.float32),
            semantic=request.semantic.to_tensor(dtype=torch.int64),
            intrinsics=request.intrinsics.to_tensor(dtype=torch.float32),
            pose=request.pose.to_tensor(dtype=torch.float32),
            frame_id=request.frame_id,
            sequence=request.sequence,
        )
        frame.validate()
        return frame

    @staticmethod
    def _resolve_backend(requested: str) -> BackendInfo:
        if requested == "auto":
            return detect_backend()
        if requested == "mlx":
            return BackendInfo(name="mlx", accelerator="metal")
        return BackendInfo(name="torch", accelerator=requested)

    @staticmethod
    def _maybe_float(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)
