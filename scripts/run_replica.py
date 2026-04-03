from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from anima_slam_gs3lam.config import load_config
from anima_slam_gs3lam.datasets.registry import build_dataset
from anima_slam_gs3lam.pipeline.slam_loop import GS3LAMLoop
from anima_slam_gs3lam.types import FrameBatch


def make_synthetic_frame(frame_id: int, image_size: tuple[int, int] = (32, 32)) -> FrameBatch:
    height, width = image_size
    rgb = torch.zeros(3, height, width, dtype=torch.float32)
    rgb[0] = 0.2 + frame_id * 0.01
    depth = torch.ones(1, height, width, dtype=torch.float32)
    semantic = torch.zeros(height, width, dtype=torch.int64)
    intrinsics = torch.tensor(
        [[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    pose = torch.eye(4, dtype=torch.float32)
    pose[0, 3] = frame_id * 0.01
    return FrameBatch(
        rgb=rgb,
        depth=depth,
        semantic=semantic,
        intrinsics=intrinsics,
        pose=pose,
        frame_id=frame_id,
        sequence="synthetic_office0",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="office0")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--synthetic-fallback", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config()
    loop = GS3LAMLoop(
        semantic_dim=config.paper_defaults.semantic.feature_dim,
        semantic_classes=config.paper_defaults.semantic.class_dim,
    )

    preset = config.dataset_presets["replica"].model_copy(update={"sequence": args.scene})
    dataset_root = Path(preset.root) / preset.sequence
    if dataset_root.exists():
        dataset = build_dataset("replica", preset)
        frame_iter = (dataset[index] for index in range(min(args.max_frames, len(dataset))))
    elif args.synthetic_fallback:
        frame_iter = (make_synthetic_frame(index) for index in range(args.max_frames))
    else:
        raise FileNotFoundError(f"Replica scene not found at {dataset_root}")

    last_step = 0
    for last_step, frame in enumerate(tqdm(frame_iter, total=args.max_frames, desc="GS3LAM Replica")):
        loop.step(frame)

    if loop.state.field is not None:
        checkpoint = loop.save_checkpoint(args.output_dir, step_idx=last_step)
        print(f"checkpoint={checkpoint}")
    else:
        print("No frames processed — no checkpoint saved.")


if __name__ == "__main__":
    main()
