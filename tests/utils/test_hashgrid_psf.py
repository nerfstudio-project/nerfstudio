"""Tests for PSF-guided hashgrid parameter selection helpers."""

from types import SimpleNamespace

import pytest
import torch

from nerfstudio.utils.hashgrid_psf import (
    calculate_axis_fwhm,
    compute_psf_guided_hashgrid_params,
    estimate_pixel_world_size,
    growth_factor_to_max_res,
    solve_growth_factor_for_target_fwhm,
)


def _make_cameras(distance: float = 2.0, focal: float = 500.0):
    camera_to_worlds = torch.eye(4, dtype=torch.float32)[:3, :].unsqueeze(0)
    camera_to_worlds[..., 2, 3] = distance
    return SimpleNamespace(
        camera_to_worlds=camera_to_worlds,
        camera_type=torch.ones((1, 1), dtype=torch.int64),
        fx=torch.tensor([[focal]], dtype=torch.float32),
        fy=torch.tensor([[focal]], dtype=torch.float32),
    )


def test_solve_growth_factor_for_target_fwhm():
    base_resolution = 16
    num_levels = 16
    target_fwhm_world = 0.01
    broadening = 3.0

    growth_factor = solve_growth_factor_for_target_fwhm(
        base_resolution=base_resolution,
        num_levels=num_levels,
        target_fwhm_world=target_fwhm_world,
        dim=3,
        broadening=broadening,
    )
    solved_fwhm = (
        calculate_axis_fwhm(
            base_resolution=base_resolution,
            growth_factor=growth_factor,
            num_levels=num_levels,
            dim=3,
        )
        * broadening
    )
    assert growth_factor > 1.0
    assert solved_fwhm == pytest.approx(target_fwhm_world, rel=1e-3, abs=1e-6)


def test_growth_factor_to_max_res():
    max_res = growth_factor_to_max_res(base_resolution=16, num_levels=16, growth_factor=1.38)
    assert max_res >= 16


def test_estimate_pixel_world_size():
    cameras = _make_cameras(distance=2.0, focal=500.0)
    pixel_world = estimate_pixel_world_size(cameras)
    assert pixel_world is not None
    assert pixel_world == pytest.approx(2.0 / 500.0, rel=1e-5, abs=1e-8)


def test_compute_psf_guided_hashgrid_params():
    scene_aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    cameras = _make_cameras(distance=2.0, focal=500.0)

    params = compute_psf_guided_hashgrid_params(
        train_cameras=cameras,
        scene_aabb=scene_aabb,
        base_resolution=16,
        num_levels=16,
        target_fwhm_pixels=0.5,
        dim=3,
        broadening=3.0,
    )
    assert params is not None
    assert params["growth_factor"] > 1.0
    assert int(params["max_res"]) >= 16
    assert params["used_scene_scale_fallback"] == 0.0

    fallback_params = compute_psf_guided_hashgrid_params(
        train_cameras=None,
        scene_aabb=scene_aabb,
        base_resolution=16,
        num_levels=16,
        target_fwhm_pixels=0.5,
        dim=3,
        broadening=3.0,
    )
    assert fallback_params is not None
    assert fallback_params["used_scene_scale_fallback"] == 1.0
