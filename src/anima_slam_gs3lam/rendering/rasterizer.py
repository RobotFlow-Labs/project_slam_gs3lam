"""CUDA-ready rasterizer wrapper with a torch fallback for Mac-safe development."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from gaussian_semantic_rasterization import (  # type: ignore
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except Exception:  # pragma: no cover - exercised via fallback tests
    GaussianRasterizationSettings = None
    GaussianRasterizer = None


@dataclass(frozen=True)
class RenderOutputs:
    rgb: torch.Tensor
    depth: torch.Tensor
    semantic_feature: torch.Tensor
    opacity: torch.Tensor


def render_field(
    field,
    *,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    near: float = 0.01,
    far: float = 100.0,
) -> RenderOutputs:
    if (
        GaussianRasterizer is not None
        and GaussianRasterizationSettings is not None
        and field.means3d.is_cuda
        and pose.is_cuda
        and intrinsics.is_cuda
    ):
        return _render_with_extension(field, pose=pose, intrinsics=intrinsics, image_size=image_size, near=near, far=far)
    return _render_with_torch_fallback(field, pose=pose, intrinsics=intrinsics, image_size=image_size)


def _render_with_extension(field, *, pose: torch.Tensor, intrinsics: torch.Tensor, image_size: tuple[int, int], near: float, far: float) -> RenderOutputs:
    height, width = image_size
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    cam_center = torch.inverse(pose)[:3, 3]
    viewmatrix = pose.unsqueeze(0).transpose(1, 2)
    proj = torch.tensor(
        [
            [2 * fx / width, 0.0, -(width - 2 * cx) / width, 0.0],
            [0.0, 2 * fy / height, -(height - 2 * cy) / height, 0.0],
            [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=pose.dtype,
        device=pose.device,
    ).unsqueeze(0).transpose(1, 2)
    full_proj = viewmatrix.bmm(proj)

    settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=width / (2 * fx),
        tanfovy=height / (2 * fy),
        bg=torch.zeros(3, dtype=field.means3d.dtype, device=field.means3d.device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False,
    )
    means2d = torch.zeros_like(field.means3d, requires_grad=True)
    rgb, semantic_feature, _, depth, opacity = GaussianRasterizer(raster_settings=settings)(
        means3D=field.means3d,
        colors_precomp=field.rgb,
        sh_objs=field.semantic_features.unsqueeze(1),
        rotations=field.normalized_quaternions(),
        opacities=field.opacities(),
        scales=field.scales(),
        means2D=means2d,
    )
    return RenderOutputs(rgb=rgb, depth=depth.unsqueeze(0), semantic_feature=semantic_feature, opacity=opacity.unsqueeze(0))


def _render_with_torch_fallback(field, *, pose: torch.Tensor, intrinsics: torch.Tensor, image_size: tuple[int, int]) -> RenderOutputs:
    height, width = image_size
    device = field.means3d.device
    dtype = field.means3d.dtype

    homogeneous = torch.cat(
        [field.means3d, torch.ones(field.num_gaussians, 1, device=device, dtype=dtype)],
        dim=-1,
    )
    camera_points = (pose @ homogeneous.T).T[:, :3]
    z = camera_points[:, 2].clamp_min(1e-6)
    x = camera_points[:, 0] / z
    y = camera_points[:, 1] / z
    u = torch.round(intrinsics[0, 0] * x + intrinsics[0, 2]).long()
    v = torch.round(intrinsics[1, 1] * y + intrinsics[1, 2]).long()

    rgb = torch.zeros(3, height, width, dtype=dtype, device=device)
    depth = torch.full((1, height, width), torch.inf, dtype=dtype, device=device)
    opacity = torch.zeros(1, height, width, dtype=dtype, device=device)
    semantic = torch.zeros(field.semantic_dim, height, width, dtype=dtype, device=device)

    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not torch.any(valid):
        depth.zero_()
        return RenderOutputs(rgb=rgb, depth=depth, semantic_feature=semantic, opacity=opacity)

    for index in torch.nonzero(valid, as_tuple=False).flatten():
        row = int(v[index].item())
        col = int(u[index].item())
        candidate_depth = z[index]
        current_depth = depth[0, row, col]
        if candidate_depth < current_depth:
            alpha = field.opacities()[index, 0]
            rgb[:, row, col] = field.rgb[index] * alpha
            semantic[:, row, col] = field.semantic_features[index] * alpha
            depth[0, row, col] = candidate_depth
            opacity[0, row, col] = alpha

    depth[torch.isinf(depth)] = 0.0
    return RenderOutputs(rgb=rgb, depth=depth, semantic_feature=semantic, opacity=opacity)
