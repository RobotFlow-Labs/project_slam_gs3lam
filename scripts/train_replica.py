"""GS3LAM Replica benchmark runner with full checkpointing and eval.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/train_replica.py \
        --scene office0 --max-frames -1

Runs the online SLAM loop on a Replica scene, saves checkpoints to
/mnt/artifacts-datai/, and reports evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from anima_slam_gs3lam.config import load_config
from anima_slam_gs3lam.datasets.registry import build_dataset
from anima_slam_gs3lam.eval.rendering import rendering_metrics, aggregate_rendering_metrics
from anima_slam_gs3lam.eval.tracking import ate_rmse_cm
from anima_slam_gs3lam.pipeline.slam_loop import GS3LAMLoop
from anima_slam_gs3lam.version import __version__

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT = "project_slam_gs3lam"
ARTIFACTS = Path("/mnt/artifacts-datai")
CKPT_DIR = ARTIFACTS / "checkpoints" / PROJECT
LOG_DIR = ARTIFACTS / "logs" / PROJECT
REPORT_DIR = ARTIFACTS / "reports" / PROJECT


def main() -> None:
    parser = argparse.ArgumentParser(description="GS3LAM Replica benchmark")
    parser.add_argument("--scene", default="office0", help="Replica scene name")
    parser.add_argument("--max-frames", type=int, default=-1, help="-1 for all frames")
    parser.add_argument("--ckpt-every", type=int, default=100, help="Checkpoint every N frames")
    parser.add_argument("--eval-every", type=int, default=5, help="Eval metrics every N frames")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--config", default="configs/default.toml", help="config file path")
    args = parser.parse_args()

    for d in [CKPT_DIR, LOG_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    runtime = config.build_runtime_config(
        "replica",
        sequence_override=args.scene,
    )

    device = torch.device(args.device)
    logger.info("[CONFIG] Scene: %s", args.scene)
    logger.info("[CONFIG] Tracking iters: %d, Mapping iters: %d",
                runtime.tracking.iterations, runtime.mapping.iterations)
    logger.info("[CONFIG] Device: %s", device)
    logger.info("[CONFIG] Version: %s", __version__)

    # Load dataset
    dataset = build_dataset("replica", runtime.dataset)
    total_frames = len(dataset) if args.max_frames < 0 else min(args.max_frames, len(dataset))
    logger.info("[DATA] %s frames in scene %s, processing %d",
                len(dataset), args.scene, total_frames)

    # GPU info
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("[GPU] %s (%.1f GB)", gpu_name, gpu_mem)

    # Build loss weight dicts from config
    track_w = runtime.tracking.loss_weights
    map_w = runtime.mapping.loss_weights
    map_lr = runtime.mapping.learning_rates

    # Initialize SLAM loop with config-driven parameters
    loop = GS3LAMLoop(
        semantic_dim=config.paper_defaults.semantic.feature_dim,
        semantic_classes=config.paper_defaults.semantic.class_dim,
        keyframe_window=runtime.mapping.mapping_window_size,
        tracking_iterations=runtime.tracking.iterations,
        mapping_iterations=runtime.mapping.iterations,
        tracking_rotation_lr=runtime.tracking.pose_rotation_lr,
        tracking_translation_lr=runtime.tracking.pose_translation_lr,
        tracking_loss_weights={
            "color": track_w.color, "depth": track_w.depth, "semantic": track_w.semantic,
        },
        mapping_loss_weights={
            "color": map_w.color, "depth": map_w.depth, "semantic": map_w.semantic,
            "big_scale": map_w.big_scale, "small_scale": map_w.small_scale,
        },
        mapping_lr={
            "means3d": map_lr.means3d,
            "rgb_colors": map_lr.rgb_colors,
            "unnorm_rotations": map_lr.unnorm_rotations,
            "logit_opacities": map_lr.logit_opacities,
            "log_scales": map_lr.log_scales,
            "obj_dc": map_lr.obj_dc,
        },
        device=device,
    )

    # Run SLAM loop
    render_metrics_list = []
    gt_poses = []
    frame_times = []
    history: list[dict] = []

    logger.info("[TRAIN] Starting SLAM loop...")
    t0 = time.time()

    for frame_idx in tqdm(range(total_frames), desc=f"GS3LAM {args.scene}"):
        frame = dataset[frame_idx]
        # frame.to(device) is handled inside loop.step()
        frame_start = time.perf_counter()
        result = loop.step(frame)
        frame_end = time.perf_counter()
        frame_ms = (frame_end - frame_start) * 1000.0
        frame_times.append(frame_ms)

        gt_poses.append(frame.pose.clone())

        entry = {
            "frame": frame_idx,
            "frame_ms": round(frame_ms, 2),
            "new_gaussians": int(result["new_gaussians"]),
            "total_gaussians": int(result.get("total_gaussians", 0)),
        }
        if "tracking_loss" in result:
            entry["tracking_loss"] = float(result["tracking_loss"])
        if "mapping_loss" in result:
            entry["mapping_loss"] = float(result["mapping_loss"])

        # Eval every N frames
        if frame_idx > 0 and frame_idx % args.eval_every == 0:
            with torch.no_grad():
                frame_dev = frame.to(device)
                render = loop.state.field(
                    pose=frame_dev.pose,
                    intrinsics=frame_dev.intrinsics,
                    image_size=tuple(frame_dev.depth.shape[-2:]),
                )
                rm = rendering_metrics(render.rgb.cpu(), frame.rgb, render.depth.cpu(), frame.depth)
                render_metrics_list.append(rm)
                entry["psnr"] = rm.psnr
                entry["ssim"] = rm.ssim

        history.append(entry)

        if frame_idx % 50 == 0 and frame_idx > 0:
            avg_ms = sum(frame_times[-50:]) / len(frame_times[-50:])
            logger.info("[STEP %d] %.0fms/frame, N=%d gaussians, track=%.4f map=%.4f",
                        frame_idx, avg_ms,
                        result.get("total_gaussians", 0),
                        entry.get("tracking_loss", 0),
                        entry.get("mapping_loss", 0))

        # Checkpoint
        if frame_idx > 0 and frame_idx % args.ckpt_every == 0:
            ckpt_path = loop.save_checkpoint(str(CKPT_DIR / args.scene), step_idx=frame_idx)
            logger.info("[CKPT] Saved %s (N=%d gaussians)", ckpt_path,
                        loop.state.field.num_gaussians)

    elapsed = time.time() - t0
    logger.info("[DONE] %d frames in %.1fs (%.1f FPS)",
                total_frames, elapsed, total_frames / elapsed if elapsed > 0 else 0)

    # Final checkpoint
    final_ckpt = loop.save_checkpoint(str(CKPT_DIR / args.scene), step_idx=total_frames - 1)
    logger.info("[CKPT] Final checkpoint: %s", final_ckpt)

    # Compute aggregate metrics
    agg_render = aggregate_rendering_metrics(render_metrics_list)
    est_poses_t = torch.stack([p.cpu() for p in loop.state.poses])
    gt_poses_t = torch.stack(gt_poses)
    ate = ate_rmse_cm(est_poses_t, gt_poses_t)
    avg_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

    logger.info("[METRICS] PSNR=%.2f SSIM=%.4f LPIPS=%.4f ATE=%.2f cm FPS=%.1f",
                agg_render.psnr, agg_render.ssim, agg_render.lpips, ate, avg_fps)

    # Save eval report
    eval_report = {
        "scene": args.scene,
        "total_frames": total_frames,
        "elapsed_s": round(elapsed, 1),
        "num_gaussians": loop.state.field.num_gaussians,
        "replica_psnr": round(agg_render.psnr, 4),
        "replica_ssim": round(agg_render.ssim, 4),
        "replica_lpips": round(agg_render.lpips, 4),
        "replica_ate_cm": round(ate, 4),
        "replica_fps": round(avg_fps, 1),
        "checkpoint": str(final_ckpt),
    }
    report_path = REPORT_DIR / f"eval_{args.scene}.json"
    report_path.write_text(json.dumps(eval_report, indent=2) + "\n")
    logger.info("[REPORT] %s", report_path)

    # Save training history
    hist_path = LOG_DIR / f"history_{args.scene}.jsonl"
    with hist_path.open("w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
    logger.info("[LOG] %s", hist_path)


if __name__ == "__main__":
    main()
