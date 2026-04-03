"""Convenience entrypoint for serving the FastAPI app."""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("anima_slam_gs3lam.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
