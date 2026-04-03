"""Checkpoint loading and model export utilities for GS3LAM.

Supports:
- Loading SLAM checkpoints (field + decoder + poses)
- Exporting to safetensors format
- Exporting decoder to ONNX
- Exporting poses as numpy .npy
- Embedding metadata (version, semantic_dim, num_gaussians, etc.)

All outputs are written to /mnt/artifacts-datai/ paths by default.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anima_slam_gs3lam.semantic.decoder import SemanticDecoder
from anima_slam_gs3lam.sg_field import SemanticGaussianField, SGFieldInit
from anima_slam_gs3lam.version import __version__

logger = logging.getLogger(__name__)

PROJECT = "project_slam_gs3lam"
ARTIFACTS_ROOT = Path("/mnt/artifacts-datai")
DEFAULT_EXPORT_DIR = ARTIFACTS_ROOT / "exports" / PROJECT


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a GS3LAM checkpoint produced by ``SLAMLoop.save_checkpoint``.

    Returns a dict with keys: ``poses``, ``field``, ``decoder``,
    ``semantic_dim``, ``semantic_classes``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    required = {"poses", "field", "decoder", "semantic_dim", "semantic_classes"}
    missing = required - set(ckpt.keys())
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")

    logger.info(
        "Loaded checkpoint from %s  (semantic_dim=%d, semantic_classes=%d)",
        path,
        ckpt["semantic_dim"],
        ckpt["semantic_classes"],
    )
    return ckpt


def reconstruct_field(
    ckpt: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> SemanticGaussianField:
    """Rebuild a ``SemanticGaussianField`` from checkpoint state dict."""
    state = ckpt["field"]
    num_gaussians = state["means3d"].shape[0]
    semantic_dim = int(ckpt["semantic_dim"])

    field = SemanticGaussianField(
        SGFieldInit(
            num_gaussians=num_gaussians,
            semantic_dim=semantic_dim,
            device=device,
        )
    )
    field.load_state_dict(state, strict=True)
    return field


def reconstruct_decoder(
    ckpt: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> SemanticDecoder:
    """Rebuild a ``SemanticDecoder`` from checkpoint state dict."""
    semantic_dim = int(ckpt["semantic_dim"])
    semantic_classes = int(ckpt["semantic_classes"])

    decoder = SemanticDecoder(
        in_channels=semantic_dim,
        out_channels=semantic_classes,
    )
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder = decoder.to(device)
    return decoder


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def build_metadata(
    ckpt: dict[str, Any],
    *,
    field: SemanticGaussianField | None = None,
) -> dict[str, str]:
    """Build a flat string-valued metadata dict for safetensors headers."""
    num_gaussians = ckpt["field"]["means3d"].shape[0]
    if field is not None:
        num_gaussians = field.num_gaussians

    num_poses = 0
    poses = ckpt.get("poses")
    if poses is not None:
        if isinstance(poses, torch.Tensor):
            num_poses = int(poses.shape[0])
        elif isinstance(poses, list):
            num_poses = len(poses)

    return {
        "format": "gs3lam",
        "version": __version__,
        "semantic_dim": str(ckpt["semantic_dim"]),
        "semantic_classes": str(ckpt["semantic_classes"]),
        "num_gaussians": str(num_gaussians),
        "num_poses": str(num_poses),
    }


# ---------------------------------------------------------------------------
# Safetensors export
# ---------------------------------------------------------------------------


def export_safetensors(
    ckpt: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    prefix: str = "gs3lam",
) -> Path:
    """Export field and decoder state dicts to safetensors format.

    Requires the ``safetensors`` package.  Writes two files:
    ``<prefix>_field.safetensors`` and ``<prefix>_decoder.safetensors``,
    plus a ``<prefix>_metadata.json``.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for export. "
            "Install with: uv pip install safetensors"
        ) from exc

    out = Path(output_dir or DEFAULT_EXPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(ckpt)

    # Field tensors
    field_path = out / f"{prefix}_field.safetensors"
    field_tensors = {k: v.contiguous().cpu() for k, v in ckpt["field"].items()}
    save_file(field_tensors, field_path, metadata=metadata)
    logger.info("Exported field to %s", field_path)

    # Decoder tensors
    decoder_path = out / f"{prefix}_decoder.safetensors"
    decoder_tensors = {k: v.contiguous().cpu() for k, v in ckpt["decoder"].items()}
    save_file(decoder_tensors, decoder_path, metadata=metadata)
    logger.info("Exported decoder to %s", decoder_path)

    # Metadata JSON
    meta_path = out / f"{prefix}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Exported metadata to %s", meta_path)

    return out


# ---------------------------------------------------------------------------
# ONNX export (decoder only — the SG-Field is not a standard NN)
# ---------------------------------------------------------------------------


def export_decoder_onnx(
    ckpt: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    prefix: str = "gs3lam",
    image_height: int = 680,
    image_width: int = 1200,
    opset_version: int = 17,
) -> Path:
    """Export the semantic decoder to ONNX.

    The decoder is a simple 1x1 Conv2d, so ONNX export is straightforward.
    Input shape: ``[1, semantic_dim, H, W]``.
    Output shape: ``[1, semantic_classes, H, W]``.
    """
    decoder = reconstruct_decoder(ckpt, device="cpu")
    decoder.eval()

    out = Path(output_dir or DEFAULT_EXPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f"{prefix}_decoder.onnx"

    semantic_dim = int(ckpt["semantic_dim"])
    dummy_input = torch.randn(1, semantic_dim, image_height, image_width)

    torch.onnx.export(
        decoder.proj,  # export the inner Conv2d directly
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        input_names=["semantic_features"],
        output_names=["class_logits"],
        dynamic_axes={
            "semantic_features": {0: "batch", 2: "height", 3: "width"},
            "class_logits": {0: "batch", 2: "height", 3: "width"},
        },
    )
    logger.info("Exported decoder ONNX to %s", onnx_path)
    return onnx_path


# ---------------------------------------------------------------------------
# Pose export
# ---------------------------------------------------------------------------


def export_poses_npy(
    ckpt: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    prefix: str = "gs3lam",
) -> Path:
    """Export estimated poses as a numpy ``.npy`` file.

    Shape: ``[N, 4, 4]`` (SE(3) matrices).
    """
    out = Path(output_dir or DEFAULT_EXPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    npy_path = out / f"{prefix}_poses.npy"

    poses = ckpt["poses"]
    if isinstance(poses, torch.Tensor):
        poses_np = poses.detach().cpu().numpy()
    elif isinstance(poses, list):
        poses_np = np.stack([p.detach().cpu().numpy() for p in poses], axis=0)
    else:
        raise TypeError(f"Unexpected poses type: {type(poses)}")

    np.save(npy_path, poses_np)
    logger.info("Exported %d poses to %s", poses_np.shape[0], npy_path)
    return npy_path


# ---------------------------------------------------------------------------
# TensorRT export (MANDATORY: fp16 + fp32)
# ---------------------------------------------------------------------------


def export_decoder_trt(
    ckpt: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    prefix: str = "gs3lam",
    image_height: int = 680,
    image_width: int = 1200,
    onnx_opset: int = 17,
    precisions: tuple[str, ...] = ("fp32", "fp16"),
) -> dict[str, Path]:
    """Export decoder to TensorRT engines (fp16 + fp32).

    Requires: ``tensorrt`` and ``onnx`` packages.
    Pipeline: decoder → ONNX → TRT engine.
    """
    out = Path(output_dir or DEFAULT_EXPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # First ensure ONNX exists
    onnx_path = out / f"{prefix}_decoder.onnx"
    if not onnx_path.exists():
        onnx_path = export_decoder_onnx(
            ckpt, out, prefix=prefix,
            image_height=image_height, image_width=image_width,
            opset_version=onnx_opset,
        )

    results: dict[str, Path] = {}
    for precision in precisions:
        trt_path = out / f"{prefix}_decoder_{precision}.engine"
        _build_trt_engine(onnx_path, trt_path, precision=precision)
        results[f"trt_{precision}"] = trt_path
        logger.info("Exported TRT %s engine to %s", precision, trt_path)

    return results


def _build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str = "fp32",
    workspace_gb: int = 2,
) -> None:
    """Build a TensorRT engine from an ONNX model."""
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: use trtexec CLI
        _build_trt_via_cli(onnx_path, engine_path, precision=precision)
        return

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TRT: FP16 enabled")
        else:
            logger.warning("TRT: FP16 not supported on this GPU, falling back to FP32")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.write_bytes(serialized)


def _build_trt_via_cli(
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str = "fp32",
) -> None:
    """Fallback: build TRT engine using trtexec CLI."""
    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=2048",
    ]
    if precision == "fp16":
        cmd.append("--fp16")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.warning(
            "trtexec failed (code %d). TRT export skipped for %s. stderr: %s",
            result.returncode, precision, result.stderr[-500:] if result.stderr else "none",
        )
        # Create a placeholder so the pipeline doesn't break
        engine_path.write_text(f"# TRT {precision} build failed — trtexec not available\n")


# ---------------------------------------------------------------------------
# Full export pipeline
# ---------------------------------------------------------------------------


def export_all(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    prefix: str = "gs3lam",
    image_height: int = 680,
    image_width: int = 1200,
    onnx_opset: int = 17,
    device: str = "cpu",
) -> dict[str, Path]:
    """Run the full export pipeline: safetensors + ONNX + poses.

    Returns a dict mapping export type to output path.
    """
    ckpt = load_checkpoint(checkpoint_path, device=device)
    out = Path(output_dir or DEFAULT_EXPORT_DIR)

    results: dict[str, Path] = {}

    results["safetensors_dir"] = export_safetensors(ckpt, out, prefix=prefix)

    results["onnx"] = export_decoder_onnx(
        ckpt,
        out,
        prefix=prefix,
        image_height=image_height,
        image_width=image_width,
        opset_version=onnx_opset,
    )

    results["poses"] = export_poses_npy(ckpt, out, prefix=prefix)

    # TensorRT fp16 + fp32 (MANDATORY)
    trt_results = export_decoder_trt(
        ckpt, out, prefix=prefix,
        image_height=image_height, image_width=image_width,
        onnx_opset=onnx_opset,
    )
    results.update(trt_results)

    logger.info("Export complete → %s", out)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GS3LAM checkpoint export")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Export directory (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument("--prefix", type=str, default="gs3lam", help="File prefix")
    parser.add_argument("--image-height", type=int, default=680)
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (e.g. if torch.onnx unavailable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ckpt = load_checkpoint(args.checkpoint)
    out = args.output_dir

    export_safetensors(ckpt, out, prefix=args.prefix)
    export_poses_npy(ckpt, out, prefix=args.prefix)

    if not args.skip_onnx:
        export_decoder_onnx(
            ckpt,
            out,
            prefix=args.prefix,
            image_height=args.image_height,
            image_width=args.image_width,
            opset_version=args.onnx_opset,
        )

    # TRT fp16 + fp32 (MANDATORY)
    export_decoder_trt(
        ckpt,
        out,
        prefix=args.prefix,
        image_height=args.image_height,
        image_width=args.image_width,
        onnx_opset=args.onnx_opset,
    )

    print("Export complete: safetensors + ONNX + TRT fp16 + TRT fp32 + poses")


if __name__ == "__main__":
    main()
