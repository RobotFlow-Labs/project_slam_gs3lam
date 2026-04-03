"""Online GS3LAM orchestration loop with paper-faithful optimization."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import torch
from torch import nn

from anima_slam_gs3lam.losses.mapping import mapping_loss
from anima_slam_gs3lam.losses.tracking import tracking_loss
from anima_slam_gs3lam.mapping.expansion import (
    compute_unobserved_mask,
    expand_field_from_frame,
    frame_to_world_points,
    estimate_mean_sq_dist,
)
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


def _build_mapping_optimizer(
    field: SemanticGaussianField,
    decoder: SemanticDecoder,
    lr: dict[str, float],
) -> torch.optim.Adam:
    """Build optimizer over field params + decoder params with per-group LR."""
    param_groups = [
        {"params": [field.means3d], "lr": lr.get("means3d", 0.0001)},
        {"params": [field.rgb], "lr": lr.get("rgb_colors", 0.0025)},
        {"params": [field.quaternions], "lr": lr.get("unnorm_rotations", 0.001)},
        {"params": [field.logit_opacities], "lr": lr.get("logit_opacities", 0.05)},
        {"params": [field.log_scales], "lr": lr.get("log_scales", 0.001)},
        {"params": [field.semantic_features], "lr": lr.get("obj_dc", 0.0025)},
        {"params": list(decoder.parameters()), "lr": lr.get("obj_dc", 0.0025)},
    ]
    return torch.optim.Adam(param_groups, eps=1e-15)


class GS3LAMLoop:
    def __init__(
        self,
        *,
        semantic_dim: int = 16,
        semantic_classes: int = 256,
        keyframe_window: int = 5,
        tracking_iterations: int = 40,
        mapping_iterations: int = 60,
        tracking_rotation_lr: float = 0.0004,
        tracking_translation_lr: float = 0.002,
        tracking_loss_weights: dict[str, float] | None = None,
        mapping_loss_weights: dict[str, float] | None = None,
        mapping_lr: dict[str, float] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.semantic_dim = semantic_dim
        self.semantic_classes = semantic_classes
        self.keyframe_window = keyframe_window
        self.tracking_iterations = tracking_iterations
        self.mapping_iterations = mapping_iterations
        self.tracking_rotation_lr = tracking_rotation_lr
        self.tracking_translation_lr = tracking_translation_lr
        self.tracking_loss_weights = tracking_loss_weights or {
            "color": 0.5, "depth": 1.0, "semantic": 0.001,
        }
        self.mapping_loss_weights = mapping_loss_weights or {
            "color": 0.5, "depth": 1.0, "semantic": 0.01,
            "big_scale": 0.01, "small_scale": 0.001,
        }
        self.mapping_lr = mapping_lr or {}
        self.device = torch.device(device)
        self.state = LoopState(
            decoder=SemanticDecoder(
                in_channels=semantic_dim, out_channels=semantic_classes,
            ).to(self.device),
        )
        self._mapping_optimizer: torch.optim.Adam | None = None

    def _rebuild_mapping_optimizer(self) -> None:
        """Rebuild after append_gaussians invalidates param references."""
        if self.state.field is None or self.state.decoder is None:
            return
        self._mapping_optimizer = _build_mapping_optimizer(
            self.state.field, self.state.decoder, self.mapping_lr,
        )

    def bootstrap(self, frame: FrameBatch, *, max_init_points: int = 30000) -> LoopState:
        frame = frame.to(self.device)
        depth_mask = frame.depth[0] > 0
        world_points, colors = frame_to_world_points(frame, depth_mask)

        # Subsample if too many initial points (L4 23GB can handle ~200K comfortably)
        n = world_points.shape[0]
        if n > max_init_points:
            indices = torch.randperm(n, device=world_points.device)[:max_init_points]
            world_points = world_points[indices]
            colors = colors[indices]

        homogeneous = torch.cat(
            [world_points, torch.ones(world_points.shape[0], 1, device=self.device, dtype=world_points.dtype)],
            dim=-1,
        )
        camera_points = torch.linalg.solve(frame.pose, homogeneous.T).T[:, :3]
        mean_sq_dist = estimate_mean_sq_dist(camera_points, frame.intrinsics)
        self.state.field = SemanticGaussianField.from_point_cloud(
            world_points, colors,
            semantic_dim=self.semantic_dim,
            mean_sq_dist=mean_sq_dist,
        ).to(self.device)
        self.state.poses.append(frame.pose.clone())
        self.state.keyframes.append(frame)
        self._rebuild_mapping_optimizer()
        return self.state

    def _run_tracking(self, frame: FrameBatch, pose_init: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        """Optimize camera pose with frozen field for tracking_iterations steps."""
        # Decompose pose into learnable rotation (as quaternion) and translation
        R = pose_init[:3, :3]
        t = pose_init[:3, 3].clone()

        # Convert rotation matrix to unnormalized quaternion for optimization
        quat = _rotation_matrix_to_quaternion(R)
        opt_quat = nn.Parameter(quat)
        opt_trans = nn.Parameter(t)

        optimizer = torch.optim.Adam([
            {"params": [opt_quat], "lr": self.tracking_rotation_lr},
            {"params": [opt_trans], "lr": self.tracking_translation_lr},
        ], eps=1e-15)

        image_size = tuple(frame.depth.shape[-2:])
        best_loss = float("inf")
        best_pose = pose_init.clone()

        # Freeze field during tracking
        for param in self.state.field.parameters():
            param.requires_grad_(False)

        for _ in range(self.tracking_iterations):
            optimizer.zero_grad()
            pose = _quaternion_translation_to_pose(opt_quat, opt_trans)
            render = self.state.field(pose=pose, intrinsics=frame.intrinsics, image_size=image_size)
            logits = self.state.decoder(render.semantic_feature)
            total, terms = tracking_loss(
                render, frame.rgb, frame.depth, logits, frame.semantic,
                weights=self.tracking_loss_weights,
            )
            if total.requires_grad:
                total.backward()
                optimizer.step()
            loss_val = total.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_pose = pose.detach().clone()

        # Unfreeze field
        for param in self.state.field.parameters():
            param.requires_grad_(True)

        return best_pose, {"tracking_loss": best_loss, **{f"track_{k}": v.item() for k, v in terms.items()}}

    def _run_mapping(self, frames: list[FrameBatch]) -> dict[str, object]:
        """Optimize field + decoder with frozen poses for mapping_iterations steps."""
        if self._mapping_optimizer is None:
            self._rebuild_mapping_optimizer()
        optimizer = self._mapping_optimizer

        last_terms: dict[str, object] = {}
        for _ in range(self.mapping_iterations):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            for kf in frames:
                pose_idx = min(kf.frame_id, len(self.state.poses) - 1)
                pose = self.state.poses[pose_idx].detach()
                image_size = tuple(kf.depth.shape[-2:])
                render = self.state.field(pose=pose, intrinsics=kf.intrinsics, image_size=image_size)
                logits = self.state.decoder(render.semantic_feature)
                loss, terms = mapping_loss(
                    render, kf.rgb, kf.depth, logits, kf.semantic,
                    self.state.field.scales(),
                    weights=self.mapping_loss_weights,
                )
                total_loss = total_loss + loss
                last_terms = {f"map_{k}": v.item() for k, v in terms.items()}

            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()

        last_terms["mapping_loss"] = total_loss.item()
        return last_terms

    def step(self, frame: FrameBatch) -> dict[str, object]:
        frame = frame.to(self.device)
        if self.state.field is None or self.state.decoder is None:
            self.bootstrap(frame)
            return {"bootstrapped": True, "pose": self.state.poses[-1], "new_gaussians": self.state.field.num_gaussians}

        # 1. Tracking: optimize pose with frozen field
        prev_pose = self.state.poses[-1]
        prev_prev_pose = self.state.poses[-2] if len(self.state.poses) > 1 else None
        pose_init = initialize_pose(prev_pose, prev_prev_pose)
        optimized_pose, track_metrics = self._run_tracking(frame, pose_init)
        self.state.poses.append(optimized_pose)

        # 2. Expansion: add Gaussians for unobserved regions
        image_size = tuple(frame.depth.shape[-2:])
        with torch.no_grad():
            render = self.state.field(pose=optimized_pose, intrinsics=frame.intrinsics, image_size=image_size)
        unobserved_mask = compute_unobserved_mask(render.opacity[0], render.depth[0], frame.depth[0])
        added = expand_field_from_frame(self.state.field, frame, unobserved_mask)
        if added > 0:
            self._rebuild_mapping_optimizer()

        # 3. Keyframe management
        self.state.keyframes.append(frame)
        self.state.keyframes = sample_keyframes(
            self.state.keyframes,
            step_idx=len(self.state.poses) - 1,
            t_opt=self.keyframe_window,
        )

        # 4. Mapping: optimize field + decoder with frozen poses
        map_metrics = self._run_mapping(self.state.keyframes)

        # 5. Pruning: remove low-opacity Gaussians every 20 frames + hard cap at 300K
        pruned = 0
        step_num = len(self.state.poses)
        if step_num > 1 and step_num % 20 == 0:
            pruned = self.state.field.prune_low_opacity(threshold=0.05)
            if pruned > 0:
                self._rebuild_mapping_optimizer()
        # Hard cap: if still over 300K, prune lowest opacity until under
        max_gaussians = 150000
        if self.state.field.num_gaussians > max_gaussians:
            opacities = self.state.field.opacities().squeeze(-1)
            _, sorted_idx = opacities.sort()
            n_remove = self.state.field.num_gaussians - max_gaussians
            remove_mask = torch.zeros(self.state.field.num_gaussians, dtype=torch.bool, device=self.device)
            remove_mask[sorted_idx[:n_remove]] = True
            pruned += self.state.field.prune(remove_mask)
            self._rebuild_mapping_optimizer()

        return {
            "bootstrapped": False,
            "pose": optimized_pose,
            "new_gaussians": added,
            "pruned_gaussians": pruned,
            "total_gaussians": self.state.field.num_gaussians,
            **track_metrics,
            **map_metrics,
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
                "step_idx": step_idx,
            },
            checkpoint_path,
        )
        self.state.checkpoints.append(checkpoint_path)
        return checkpoint_path


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return torch.stack([w, x, y, z]).to(dtype=R.dtype, device=R.device)


def _quaternion_translation_to_pose(quat: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] + translation to 4x4 c2w matrix."""
    q = torch.nn.functional.normalize(quat, dim=0)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])
    pose = torch.eye(4, dtype=quat.dtype, device=quat.device)
    pose[:3, :3] = R
    pose[:3, 3] = trans
    return pose
