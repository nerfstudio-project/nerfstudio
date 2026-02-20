"""PSF-guided helpers for hashgrid parameter selection."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

_PERSPECTIVE_CAMERA_TYPE = 1


def _get_resolutions(base_resolution: int, growth_factor: float, num_levels: int) -> np.ndarray:
    """Return per-level hash resolutions N_l = N_min * b^l."""
    levels = np.arange(num_levels, dtype=np.float64)
    return float(base_resolution) * (float(growth_factor) ** levels)


def calculate_axis_fwhm(
    base_resolution: int,
    growth_factor: float,
    num_levels: int,
    *,
    dim: int = 3,
) -> float:
    """Estimate axis FWHM using the revised weighted-kernel approximation."""
    resolutions = _get_resolutions(base_resolution=base_resolution, growth_factor=growth_factor, num_levels=num_levels)
    sum_dim = float(np.sum(resolutions**dim)) + 1e-20
    sum_dim_p1 = float(np.sum(resolutions ** (dim + 1))) + 1e-20
    return sum_dim / sum_dim_p1


def solve_growth_factor_for_target_fwhm(
    base_resolution: int,
    num_levels: int,
    target_fwhm_world: float,
    *,
    dim: int = 3,
    broadening: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 64,
) -> float:
    """Binary-search the hashgrid growth factor `b` for a target empirical FWHM."""
    if target_fwhm_world <= 0.0:
        raise ValueError("target_fwhm_world must be > 0")
    if base_resolution <= 0:
        raise ValueError("base_resolution must be > 0")
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if broadening <= 0.0:
        raise ValueError("broadening must be > 0")

    # Empirical PSF is wider than idealized theory by this broadening factor.
    target_fwhm_world = float(target_fwhm_world) / float(broadening)

    def fwhm_from_b(growth_factor: float) -> float:
        return calculate_axis_fwhm(
            base_resolution=base_resolution,
            growth_factor=growth_factor,
            num_levels=num_levels,
            dim=dim,
        )

    lo = 1.0 + 1e-6
    hi = lo * 2.0
    f_lo = fwhm_from_b(lo)
    if target_fwhm_world >= f_lo:
        return lo

    f_hi = fwhm_from_b(hi)
    growth_guard = 0
    while f_hi > target_fwhm_world and growth_guard < 64:
        hi *= 1.5
        f_hi = fwhm_from_b(hi)
        growth_guard += 1
        if hi > 1e6:
            raise ValueError("Unable to bracket target FWHM; try different inputs.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fwhm_from_b(mid)
        if abs(f_mid - target_fwhm_world) < tol:
            return mid
        if f_mid > target_fwhm_world:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def growth_factor_to_max_res(base_resolution: int, num_levels: int, growth_factor: float) -> int:
    """Convert growth factor `b` to integer max resolution used by Nerfstudio configs."""
    if num_levels <= 1:
        return int(base_resolution)
    max_res = float(base_resolution) * (float(growth_factor) ** float(num_levels - 1))
    return int(max(base_resolution, round(max_res)))


def estimate_pixel_world_size(cameras: Optional[Any]) -> Optional[float]:
    """Estimate world-units per pixel from perspective cameras."""
    if cameras is None:
        return None
    if cameras.camera_to_worlds.numel() == 0:
        return None

    required_attrs = ("camera_to_worlds", "camera_type", "fx", "fy")
    if not all(hasattr(cameras, attr) for attr in required_attrs):
        return None

    camera_type = cameras.camera_type.reshape(-1)
    perspective_mask = camera_type == _PERSPECTIVE_CAMERA_TYPE
    if not torch.any(perspective_mask):
        return None

    origins = cameras.camera_to_worlds[..., :3, 3].reshape(-1, 3).to(dtype=torch.float64)
    fx = cameras.fx.reshape(-1).to(dtype=torch.float64)
    fy = cameras.fy.reshape(-1).to(dtype=torch.float64)
    base_valid = perspective_mask & torch.isfinite(fx) & torch.isfinite(fy) & (fx > 0.0) & (fy > 0.0)
    if not torch.any(base_valid):
        return None

    origins = origins[base_valid]
    fx = fx[base_valid]
    fy = fy[base_valid]
    distances = torch.linalg.norm(origins, dim=-1)
    valid = torch.isfinite(distances) & (distances > 0.0)
    if not torch.any(valid):
        return None

    # Pinhole approximation: one pixel subtends z/f focal world units at depth z.
    distances = distances[valid]
    px_x = distances / fx[valid]
    px_y = distances / fy[valid]
    pixel_world = 0.5 * (px_x + px_y)
    if pixel_world.numel() == 0:
        return None
    return float(torch.median(pixel_world).item())


def estimate_scene_scale_from_aabb(aabb: torch.Tensor) -> float:
    """Fallback scene scale based on longest aabb side in world units."""
    if aabb.numel() != 6:
        return 1.0
    extents = (aabb[1] - aabb[0]).to(dtype=torch.float64)
    max_extent = float(torch.max(torch.abs(extents)).item())
    return max(max_extent, 1e-6)


def compute_psf_guided_hashgrid_params(
    *,
    train_cameras: Optional[Any],
    scene_aabb: torch.Tensor,
    base_resolution: int,
    num_levels: int,
    target_fwhm_pixels: float,
    dim: int = 3,
    broadening: float = 3.0,
) -> Optional[Dict[str, float]]:
    """Compute PSF-guided hashgrid growth/max-res from dataset scale and image resolution."""
    if target_fwhm_pixels <= 0.0:
        return None

    pixel_world = estimate_pixel_world_size(train_cameras) if train_cameras is not None else None
    used_scene_scale_fallback = pixel_world is None
    if pixel_world is None:
        pixel_world = estimate_scene_scale_from_aabb(scene_aabb)

    target_fwhm_world = float(target_fwhm_pixels) * float(pixel_world)
    if target_fwhm_world <= 0.0:
        return None

    growth_factor = solve_growth_factor_for_target_fwhm(
        base_resolution=base_resolution,
        num_levels=num_levels,
        target_fwhm_world=target_fwhm_world,
        dim=dim,
        broadening=broadening,
    )
    max_res = growth_factor_to_max_res(
        base_resolution=base_resolution,
        num_levels=num_levels,
        growth_factor=growth_factor,
    )
    return {
        "growth_factor": float(growth_factor),
        "max_res": float(max_res),
        "target_fwhm_world": float(target_fwhm_world),
        "pixel_world": float(pixel_world),
        "used_scene_scale_fallback": float(1.0 if used_scene_scale_fallback else 0.0),
    }
