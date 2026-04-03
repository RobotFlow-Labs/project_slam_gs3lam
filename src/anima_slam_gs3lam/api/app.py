"""FastAPI entrypoint for the GS3LAM service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from anima_slam_gs3lam.api.schemas import (
    HealthResponse,
    LoadSceneRequest,
    LoadSceneResponse,
    SnapshotResponse,
    StepFrameRequest,
    StepFrameResponse,
)
from anima_slam_gs3lam.api.service import SessionService


service = SessionService()
app = FastAPI(
    title="ANIMA SLAM-GS3LAM",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/v1/gs3lam/load-scene", response_model=LoadSceneResponse)
def load_scene(request: LoadSceneRequest) -> LoadSceneResponse:
    return service.load_scene(request)


@app.post("/v1/gs3lam/step-frame", response_model=StepFrameResponse)
def step_frame(request: StepFrameRequest) -> StepFrameResponse:
    return service.step(request)


@app.get("/v1/gs3lam/snapshot/{session_id}", response_model=SnapshotResponse)
def snapshot(session_id: str) -> SnapshotResponse:
    try:
        return service.snapshot(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
