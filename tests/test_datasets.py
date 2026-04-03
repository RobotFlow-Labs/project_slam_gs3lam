from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from anima_slam_gs3lam.config import default_config
from anima_slam_gs3lam.datasets.registry import build_dataset


def _write_rgb(path: Path, size: tuple[int, int]) -> None:
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[..., 0] = 120
    Image.fromarray(array, mode="RGB").save(path)


def _write_depth(path: Path, size: tuple[int, int], value: int = 1000) -> None:
    array = np.full((size[1], size[0]), value, dtype=np.uint16)
    Image.fromarray(array).save(path)


def _write_semantic(path: Path, size: tuple[int, int], label: int = 3) -> None:
    array = np.full((size[1], size[0]), label, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_pose_txt(path: Path) -> None:
    np.savetxt(path, np.eye(4, dtype=np.float32))


def test_replica_dataset_contract(tmp_path: Path):
    scene_root = tmp_path / "Replica" / "office0"
    (scene_root / "results").mkdir(parents=True)
    (scene_root / "semantic_class").mkdir(parents=True)
    _write_rgb(scene_root / "results" / "frame0000.jpg", (32, 16))
    _write_depth(scene_root / "results" / "depth0000.png", (32, 16), value=6554)
    _write_semantic(scene_root / "semantic_class" / "semantic_class_0000.png", (32, 16))
    (scene_root / "traj.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")

    config = default_config()
    preset = config.dataset_presets["replica"].model_copy(
        update={"root": str(tmp_path / "Replica"), "desired_image_height": 16, "desired_image_width": 32}
    )
    dataset = build_dataset("replica", preset)
    batch = dataset[0]
    assert batch.rgb.shape == (3, 16, 32)
    assert batch.depth.shape == (1, 16, 32)
    assert batch.semantic.shape == (16, 32)


def test_scannet_dataset_contract(tmp_path: Path):
    scene_root = tmp_path / "scannet" / "scene0059_00"
    for folder in ("color", "depth", "label-filt", "pose"):
        (scene_root / folder).mkdir(parents=True)
    _write_rgb(scene_root / "color" / "0.jpg", (24, 12))
    _write_depth(scene_root / "depth" / "0.png", (24, 12))
    _write_semantic(scene_root / "label-filt" / "0.png", (24, 12))
    _write_pose_txt(scene_root / "pose" / "0.txt")

    config = default_config()
    preset = config.dataset_presets["scannet"].model_copy(
        update={"root": str(tmp_path / "scannet"), "desired_image_height": 12, "desired_image_width": 24}
    )
    dataset = build_dataset("scannet", preset)
    batch = dataset[0]
    assert batch.pose.shape == (4, 4)
    assert batch.sequence == "scene0059_00"


def test_tum_dataset_contract(tmp_path: Path):
    scene_root = tmp_path / "TUM-DEVA" / "rgbd_dataset_freiburg1_desk"
    (scene_root / "rgb").mkdir(parents=True)
    (scene_root / "depth").mkdir(parents=True)
    (scene_root / "object_mask").mkdir(parents=True)
    _write_rgb(scene_root / "rgb" / "0001.png", (20, 10))
    _write_depth(scene_root / "depth" / "0001.png", (20, 10), value=5000)
    _write_semantic(scene_root / "object_mask" / "0001.png", (20, 10), label=7)
    (scene_root / "rgb.txt").write_text("0.0 rgb/0001.png\n")
    (scene_root / "depth.txt").write_text("0.0 depth/0001.png\n")
    (scene_root / "groundtruth.txt").write_text("timestamp tx ty tz qx qy qz qw\n0.0 0 0 0 0 0 0 1\n")

    config = default_config()
    preset = config.dataset_presets["tum"].model_copy(
        update={"root": str(tmp_path / "TUM-DEVA"), "desired_image_height": 10, "desired_image_width": 20}
    )
    dataset = build_dataset("tum", preset)
    batch = dataset[0]
    assert batch.rgb.shape == (3, 10, 20)
    assert batch.semantic.max().item() == 7
