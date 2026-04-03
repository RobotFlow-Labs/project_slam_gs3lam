"""Typed configuration for paper-faithful GS3LAM scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, ConfigDict, Field

DatasetName = Literal["replica", "scannet", "tum"]


class ProjectConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "anima-slam-gs3lam"
    codename: str = "SLAM-GS3LAM"
    functional_name: str = "Gaussian Semantic Splatting SLAM"
    wave: int = 7
    paper_arxiv: str = "2603.27781"
    python_version: str = "3.11"


class ComputeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: Literal["auto", "mlx", "cuda", "cpu"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    prefer_mlx_on_mac: bool = True
    prefer_cuda_on_linux: bool = True


class DataRootsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shared_volume: str = "/Volumes/AIFlowDev/RobotFlowLabs/datasets"
    repos_volume: str = "/Volumes/AIFlowDev/RobotFlowLabs/repos/wave7"
    dataset_root_rel: str = "datasets/slam/gs3lam"
    model_root_rel: str = "models/slam/gs3lam"


class CameraConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_height: int
    image_width: int
    fx: float
    fy: float
    cx: float
    cy: float
    png_depth_scale: float
    crop_edge: int = 0


class DatasetPreset(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: DatasetName
    sequence: str
    root: str
    desired_image_height: int
    desired_image_width: int
    start: int = 0
    end: int = -1
    stride: int = 1
    num_frames: int = -1
    camera: CameraConfig


class SemanticConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_dim: int = 16
    class_dim: int = 256


class LossWeightsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    color: float = 0.5
    depth: float = 1.0
    semantic: float
    big_scale: float
    small_scale: float


class LearningRatesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    means3d: float = 0.0001
    rgb_colors: float = 0.0025
    unnorm_rotations: float = 0.001
    logit_opacities: float = 0.05
    log_scales: float = 0.001
    cam_unnorm_rots: float = 0.0
    cam_trans: float = 0.0
    obj_dc: float = 0.0025


class TrackingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    iterations: int
    pose_rotation_lr: float
    pose_translation_lr: float
    loss_weights: LossWeightsConfig


class MappingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    iterations: int
    first_frame_iterations: int = 1000
    opt_rskm_interval: int = 5
    densify_threshold: float
    map_every: int = 1
    keyframe_every: int = 5
    mapping_window_size: int
    loss_weights: LossWeightsConfig
    learning_rates: LearningRatesConfig = Field(default_factory=LearningRatesConfig)


class PaperDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)

    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    replica_tracking_iterations: int = 40
    replica_mapping_iterations: int = 60
    scannet_tracking_iterations: int = 100
    scannet_mapping_iterations: int = 30
    tum_tracking_iterations: int = 360
    tum_mapping_iterations: int = 150
    tracking_rotation_lr: float = 0.0004
    tracking_translation_lr: float = 0.002
    tracking_loss: LossWeightsConfig = Field(
        default_factory=lambda: LossWeightsConfig(
            semantic=0.001,
            big_scale=0.05,
            small_scale=0.005,
        )
    )
    mapping_loss: LossWeightsConfig = Field(
        default_factory=lambda: LossWeightsConfig(
            semantic=0.01,
            big_scale=0.01,
            small_scale=0.001,
        )
    )


class IterationOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    tracking_iterations: int
    mapping_iterations: int


class RepoOverrides(BaseModel):
    model_config = ConfigDict(frozen=True)

    scannet: IterationOverride = Field(
        default_factory=lambda: IterationOverride(
            tracking_iterations=200,
            mapping_iterations=60,
        )
    )


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: DatasetPreset
    tracking: TrackingConfig
    mapping: MappingConfig


class GS3LAMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    data: DataRootsConfig = Field(default_factory=DataRootsConfig)
    paper_defaults: PaperDefaults = Field(default_factory=PaperDefaults)
    repo_overrides: RepoOverrides = Field(default_factory=RepoOverrides)
    dataset_presets: dict[DatasetName, DatasetPreset]

    def build_runtime_config(
        self,
        dataset_name: DatasetName,
        *,
        use_repo_overrides: bool = False,
        root_override: str | None = None,
        sequence_override: str | None = None,
    ) -> RuntimeConfig:
        preset = self.dataset_presets[dataset_name]
        if root_override is not None or sequence_override is not None:
            preset = preset.model_copy(
                update={
                    "root": root_override or preset.root,
                    "sequence": sequence_override or preset.sequence,
                }
            )

        tracking_iterations, mapping_iterations = self._iteration_pair(
            dataset_name,
            use_repo_overrides=use_repo_overrides,
        )

        densify_threshold = 0.1 if dataset_name == "replica" else 0.5
        mapping_window_size = {
            "replica": 24,
            "scannet": 10,
            "tum": 20,
        }[dataset_name]

        return RuntimeConfig(
            dataset=preset,
            tracking=TrackingConfig(
                iterations=tracking_iterations,
                pose_rotation_lr=self.paper_defaults.tracking_rotation_lr,
                pose_translation_lr=self.paper_defaults.tracking_translation_lr,
                loss_weights=self.paper_defaults.tracking_loss,
            ),
            mapping=MappingConfig(
                iterations=mapping_iterations,
                opt_rskm_interval=5,
                densify_threshold=densify_threshold,
                mapping_window_size=mapping_window_size,
                loss_weights=self.paper_defaults.mapping_loss,
            ),
        )

    def _iteration_pair(
        self,
        dataset_name: DatasetName,
        *,
        use_repo_overrides: bool,
    ) -> tuple[int, int]:
        paper_defaults = {
            "replica": (
                self.paper_defaults.replica_tracking_iterations,
                self.paper_defaults.replica_mapping_iterations,
            ),
            "scannet": (
                self.paper_defaults.scannet_tracking_iterations,
                self.paper_defaults.scannet_mapping_iterations,
            ),
            "tum": (
                self.paper_defaults.tum_tracking_iterations,
                self.paper_defaults.tum_mapping_iterations,
            ),
        }
        if dataset_name == "scannet" and use_repo_overrides:
            override = self.repo_overrides.scannet
            return override.tracking_iterations, override.mapping_iterations
        return paper_defaults[dataset_name]


def default_config() -> GS3LAMConfig:
    return GS3LAMConfig(
        dataset_presets={
            "replica": DatasetPreset(
                name="replica",
                sequence="office0",
                root="/mnt/forge-data/datasets/slam/gs3lam/Replica",
                desired_image_height=680,
                desired_image_width=1200,
                camera=CameraConfig(
                    image_height=680,
                    image_width=1200,
                    fx=600.0,
                    fy=600.0,
                    cx=599.5,
                    cy=339.5,
                    png_depth_scale=6553.5,
                ),
            ),
            "scannet": DatasetPreset(
                name="scannet",
                sequence="scene0059_00",
                root="/mnt/forge-data/datasets/slam/gs3lam/scannet",
                desired_image_height=480,
                desired_image_width=640,
                camera=CameraConfig(
                    image_height=968,
                    image_width=1296,
                    fx=1169.621094,
                    fy=1167.105103,
                    cx=646.295044,
                    cy=489.927032,
                    png_depth_scale=1000.0,
                ),
            ),
            "tum": DatasetPreset(
                name="tum",
                sequence="rgbd_dataset_freiburg1_desk",
                root="/mnt/forge-data/datasets/slam/gs3lam/TUM-DEVA",
                desired_image_height=480,
                desired_image_width=640,
                camera=CameraConfig(
                    image_height=480,
                    image_width=640,
                    fx=517.3,
                    fy=516.5,
                    cx=318.6,
                    cy=255.3,
                    png_depth_scale=5000.0,
                    crop_edge=8,
                ),
            ),
        }
    )


def load_config(path: str | Path = "configs/default.toml") -> GS3LAMConfig:
    config = default_config()
    candidate = Path(path)
    if not candidate.exists():
        return config

    with candidate.open("rb") as handle:
        payload = tomllib.load(handle)

    merged = config.model_dump(mode="python")
    for section in ("project", "compute", "data", "paper_defaults", "repo_overrides"):
        if section in payload:
            merged[section].update(payload[section])

    if "dataset_presets" in payload:
        for name, preset in payload["dataset_presets"].items():
            current = merged["dataset_presets"][name]
            camera_update = preset.get("camera", {})
            current.update({key: value for key, value in preset.items() if key != "camera"})
            current["camera"].update(camera_update)

    return GS3LAMConfig(**merged)
