"""Online GS3LAM orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import torch

from anima_slam_gs3lam.losses.mapping import mapping_loss
from anima_slam_gs3lam.losses.tracking import tracking_loss
from anima_slam_gs3lam.mapping.expansion import compute_unobserved_mask, expand_field_from_frame
from anima_slam_gs3lam.mapping.rskm import sample_keyframes
from anima_slam_gs3lam.semantic.decoder import SemanticDecoder
from anima_slam_gs3lam.sg_field import SemanticGaussianField
from anima_slam_gs3lam.tracking.tracker import initialize_pose
from anima_slam_gs3lam.types import FrameBatch


@dataclass
class LoopState:
    field: SemanticGaussianField | None = None
    decoder: SemanticDecoder | None = None
    poses: list[torch.Tensor] = dc_field(default_factory=list)
    keyframes: list[FrameBatch] = dc_field(default_factory=list)
    checkpoints: list[Path] = dc_field(default_factory=list)


class GS3LAMLoop:
    def __init__(
        self,
        *,
        semantic_dim: int = 16,
        semantic_classes: int = 256,
        keyframe_window: int = 5,
    ) -> None:
        self.semantic_dim = semantic_dim
        self.semantic_classes = semantic_classes
        self.keyframe_window = keyframe_window
        self.state = LoopState(decoder=SemanticDecoder(in_channels=semantic_dim, out_channels=semantic_classes))

    def bootstrap(self, frame: FrameBatch) -> LoopState:
        depth_mask = frame.depth[0] > 0
        from anima_slam_gs3lam.mapping.expansion import frame_to_world_points, estimate_mean_sq_dist

        world_points, colors = frame_to_world_points(frame, depth_mask)
        homogeneous = torch.cat(
            [world_points, torch.ones((world_points.shape[0], 1), device=world_points.device, dtype=world_points.dtype)],
            dim=-1,
        )
        camera_points = torch.linalg.solve(frame.pose, homogeneous.T).T[:, :3]
        mean_sq_dist = estimate_mean_sq_dist(camera_points, frame.intrinsics)
        self.state.field = SemanticGaussianField.from_point_cloud(
            world_points,
            colors,
            semantic_dim=self.semantic_dim,
            mean_sq_dist=mean_sq_dist,
        )
        self.state.poses.append(frame.pose.clone())
        self.state.keyframes.append(frame)
        return self.state

    def step(self, frame: FrameBatch) -> dict[str, object]:
        if self.state.field is None or self.state.decoder is None:
            self.bootstrap(frame)
            return {"bootstrapped": True, "pose": frame.pose, "new_gaussians": self.state.field.num_gaussians}

        prev_pose = self.state.poses[-1]
        prev_prev_pose = self.state.poses[-2] if len(self.state.poses) > 1 else None
        pose_init = initialize_pose(prev_pose, prev_prev_pose)

        render = self.state.field(pose=pose_init, intrinsics=frame.intrinsics, image_size=frame.depth.shape[-2:])
        logits = self.state.decoder(render.semantic_feature)
        track_total, track_terms = tracking_loss(
            render,
            frame.rgb,
            frame.depth,
            logits,
            frame.semantic,
        )
        unobserved_mask = compute_unobserved_mask(render.opacity[0], render.depth[0], frame.depth[0])
        added = expand_field_from_frame(self.state.field, frame, unobserved_mask)
        map_total, map_terms = mapping_loss(
            render,
            frame.rgb,
            frame.depth,
            logits,
            frame.semantic,
            self.state.field.scales(),
        )

        self.state.poses.append(pose_init)
        self.state.keyframes.append(frame)
        self.state.keyframes = sample_keyframes(
            self.state.keyframes,
            step_idx=len(self.state.poses) - 1,
            t_opt=self.keyframe_window,
        )
        return {
            "bootstrapped": False,
            "pose": pose_init,
            "tracking_loss": track_total,
            "tracking_terms": track_terms,
            "mapping_loss": map_total,
            "mapping_terms": map_terms,
            "new_gaussians": added,
        }

    def save_checkpoint(self, output_dir: str | Path, *, step_idx: int) -> Path:
        if self.state.field is None or self.state.decoder is None:
            raise RuntimeError("Cannot save checkpoint before bootstrap")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / f"gs3lam_step_{step_idx:04d}.pt"
        torch.save(
            {
                "poses": self.state.poses,
                "field": self.state.field.state_dict(),
                "decoder": self.state.decoder.state_dict(),
                "semantic_dim": self.semantic_dim,
                "semantic_classes": self.semantic_classes,
            },
            checkpoint_path,
        )
        self.state.checkpoints.append(checkpoint_path)
        return checkpoint_path
