"""Runtime aggregation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeReport:
    per_iteration_ms: float
    per_frame_ms: float
    fps: float


def runtime_report(iteration_times_ms: list[float], frame_times_ms: list[float]) -> RuntimeReport:
    avg_iter = sum(iteration_times_ms) / len(iteration_times_ms) if iteration_times_ms else 0.0
    avg_frame = sum(frame_times_ms) / len(frame_times_ms) if frame_times_ms else 0.0
    fps = 1000.0 / avg_frame if avg_frame > 0 else 0.0
    return RuntimeReport(per_iteration_ms=avg_iter, per_frame_ms=avg_frame, fps=fps)
