"""Semantic Gaussian Field container for GS3LAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from anima_slam_gs3lam.rendering.rasterizer import RenderOutputs, render_field


GaussianDistribution = Literal["isotropic", "anisotropic"]


@dataclass(frozen=True)
class SGFieldInit:
    num_gaussians: int
    semantic_dim: int = 16
    distribution: GaussianDistribution = "isotropic"
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32


class SemanticGaussianField(nn.Module):
    """Paper-faithful SG-Field parameter container for `(mu, Sigma, o, c, f)`."""

    def __init__(self, init: SGFieldInit) -> None:
        super().__init__()
        self.semantic_dim = init.semantic_dim
        self.distribution = init.distribution
        self.scale_dim = 1 if init.distribution == "isotropic" else 3

        device = init.device
        shape = (init.num_gaussians,)
        means3d = torch.zeros((*shape, 3), dtype=init.dtype, device=device)
        quaternions = torch.zeros((*shape, 4), dtype=init.dtype, device=device)
        quaternions[:, 0] = 1.0
        log_scales = torch.zeros((*shape, self.scale_dim), dtype=init.dtype, device=device)
        logit_opacities = torch.zeros((*shape, 1), dtype=init.dtype, device=device)
        rgb = torch.zeros((*shape, 3), dtype=init.dtype, device=device)
        semantic_features = torch.randn(
            (*shape, init.semantic_dim),
            dtype=init.dtype,
            device=device,
        ) * 0.01

        self.means3d = nn.Parameter(means3d)
        self.quaternions = nn.Parameter(quaternions)
        self.log_scales = nn.Parameter(log_scales)
        self.logit_opacities = nn.Parameter(logit_opacities)
        self.rgb = nn.Parameter(rgb)
        self.semantic_features = nn.Parameter(semantic_features)

    @classmethod
    def from_point_cloud(
        cls,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        *,
        semantic_dim: int = 16,
        distribution: GaussianDistribution = "isotropic",
        mean_sq_dist: torch.Tensor | None = None,
    ) -> "SemanticGaussianField":
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [N, 3]")
        if xyz.shape[0] != rgb.shape[0]:
            raise ValueError("xyz and rgb must have the same number of points")

        field = cls(
            SGFieldInit(
                num_gaussians=xyz.shape[0],
                semantic_dim=semantic_dim,
                distribution=distribution,
                device=xyz.device,
                dtype=xyz.dtype,
            )
        )
        with torch.no_grad():
            field.means3d.copy_(xyz)
            field.rgb.copy_(rgb.clamp(0.0, 1.0))
            if mean_sq_dist is not None:
                if mean_sq_dist.ndim != 1 or mean_sq_dist.shape[0] != xyz.shape[0]:
                    raise ValueError("mean_sq_dist must have shape [N]")
                scales = torch.log(torch.sqrt(mean_sq_dist.clamp_min(1e-12))).unsqueeze(-1)
                if field.scale_dim == 3:
                    scales = scales.repeat(1, 3)
                field.log_scales.copy_(scales)
        return field

    @property
    def num_gaussians(self) -> int:
        return int(self.means3d.shape[0])

    def normalized_quaternions(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quaternions, dim=-1)

    def scales(self) -> torch.Tensor:
        if self.scale_dim == 1:
            return torch.exp(self.log_scales).repeat(1, 3)
        return torch.exp(self.log_scales)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacities)

    def parameter_dict(self) -> dict[str, torch.Tensor]:
        return {
            "means3d": self.means3d,
            "quaternions": self.quaternions,
            "log_scales": self.log_scales,
            "logit_opacities": self.logit_opacities,
            "rgb": self.rgb,
            "semantic_features": self.semantic_features,
        }

    def covariance_diagonal(self) -> torch.Tensor:
        return self.scales().square()

    def append_gaussians(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        semantic_features: torch.Tensor | None = None,
        mean_sq_dist: torch.Tensor | None = None,
    ) -> None:
        new_field = self.from_point_cloud(
            xyz,
            rgb,
            semantic_dim=self.semantic_dim,
            distribution=self.distribution,
            mean_sq_dist=mean_sq_dist,
        )
        if semantic_features is not None:
            if semantic_features.shape != new_field.semantic_features.shape:
                raise ValueError("semantic_features shape must match appended Gaussian count and semantic dim")
            with torch.no_grad():
                new_field.semantic_features.copy_(semantic_features)

        with torch.no_grad():
            self.means3d = nn.Parameter(torch.cat([self.means3d, new_field.means3d], dim=0))
            self.quaternions = nn.Parameter(torch.cat([self.quaternions, new_field.quaternions], dim=0))
            self.log_scales = nn.Parameter(torch.cat([self.log_scales, new_field.log_scales], dim=0))
            self.logit_opacities = nn.Parameter(torch.cat([self.logit_opacities, new_field.logit_opacities], dim=0))
            self.rgb = nn.Parameter(torch.cat([self.rgb, new_field.rgb], dim=0))
            self.semantic_features = nn.Parameter(
                torch.cat([self.semantic_features, new_field.semantic_features], dim=0)
            )

    def forward(
        self,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
    ) -> RenderOutputs:
        return render_field(self, pose=pose, intrinsics=intrinsics, image_size=image_size)
