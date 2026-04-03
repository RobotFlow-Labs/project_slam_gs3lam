"""FastAPI entrypoint for the GS3LAM service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

import time

import torch

from anima_slam_gs3lam.api.schemas import (
    HealthResponse,
    InfoResponse,
    LoadSceneRequest,
    LoadSceneResponse,
    ReadyResponse,
    SnapshotResponse,
    StepFrameRequest,
    StepFrameResponse,
)
from anima_slam_gs3lam.api.service import SessionService
from anima_slam_gs3lam.version import __version__

_start_time = time.monotonic()
service = SessionService()
app = FastAPI(
    title="ANIMA SLAM-GS3LAM",
    version=__version__,
)


# ── ANIMA standard endpoints ──


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/ready")
def ready() -> ReadyResponse:
    return ReadyResponse(
        ready=True,
        version=__version__,
        weights_loaded=len(service.sessions) > 0,
    )


@app.get("/info")
def info() -> InfoResponse:
    return InfoResponse(
        version=__version__,
        gpu_available=torch.cuda.is_available(),
        active_sessions=len(service.sessions),
    )


# ── GS3LAM-specific endpoints ──


@app.post("/v1/gs3lam/load-scene", response_model=LoadSceneResponse)
def load_scene(request: LoadSceneRequest) -> LoadSceneResponse:
    return service.load_scene(request)


@app.post("/v1/gs3lam/step-frame", response_model=StepFrameResponse)
@app.post("/predict", response_model=StepFrameResponse)
def step_frame(request: StepFrameRequest) -> StepFrameResponse:
    return service.step(request)


@app.get("/v1/gs3lam/snapshot/{session_id}", response_model=SnapshotResponse)
def snapshot(session_id: str) -> SnapshotResponse:
    try:
        return service.snapshot(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
