"""Convenience entrypoint for serving the FastAPI app."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("ANIMA_SERVE_PORT", "8080"))
    uvicorn.run("anima_slam_gs3lam.api.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
