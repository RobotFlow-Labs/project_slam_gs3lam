"""ANIMA serve entrypoint for GS3LAM.

Supports three modes:
  - ``python -m anima_slam_gs3lam``           → FastAPI server
  - ``python -m anima_slam_gs3lam.serve``     → FastAPI server
  - Docker CMD                                → FastAPI server

Environment variables:
  ANIMA_SERVE_PORT   — port (default 8080)
  ANIMA_DEVICE       — auto|cuda|cpu (default auto)
  ANIMA_MODULE_NAME  — module identity
"""

from __future__ import annotations

import logging
import os

import uvicorn

logger = logging.getLogger(__name__)


def main() -> None:
    port = int(os.environ.get("ANIMA_SERVE_PORT", "8080"))
    module_name = os.environ.get("ANIMA_MODULE_NAME", "slam-gs3lam")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s] [{module_name}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting %s on port %d", module_name, port)

    uvicorn.run(
        "anima_slam_gs3lam.api.app:app",
        host="0.0.0.0",  # noqa: S104
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
