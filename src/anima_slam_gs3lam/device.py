"""Runtime backend detection for local Mac development and future CUDA serving."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInfo:
    name: str
    accelerator: str


def _torch_backend() -> BackendInfo:
    import torch

    if torch.cuda.is_available():
        return BackendInfo(name="torch", accelerator="cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return BackendInfo(name="torch", accelerator="mps")
    return BackendInfo(name="torch", accelerator="cpu")


def detect_backend() -> BackendInfo:
    requested = os.environ.get("ANIMA_BACKEND", "auto").lower()
    if requested in {"cpu", "cuda", "mps"}:
        return BackendInfo(name="torch", accelerator=requested)
    if requested == "mlx":
        return BackendInfo(name="mlx", accelerator="metal")

    try:
        import mlx.core as mx  # noqa: F401

        return BackendInfo(name="mlx", accelerator="metal")
    except Exception:
        return _torch_backend()


def get_backend() -> str:
    return detect_backend().accelerator


def get_device():
    backend = detect_backend()
    if backend.name == "mlx":
        import mlx.core as mx

        return mx.default_device()

    import torch

    return torch.device(backend.accelerator)
