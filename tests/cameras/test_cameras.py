"""
Test the camera classes.
"""

import torch

from nerfactory.cameras.cameras import EquirectangularCamera, PinholeCamera
from nerfactory.cameras.rays import RayBundle


def test_pinhole_camera():
    """Test that the pinhole camera model works."""
    pinhole_camera = PinholeCamera(cx=400.0, cy=400.0, fx=10.0, fy=10.0)
    camera_ray_bundle = pinhole_camera.get_camera_ray_bundle()
    assert isinstance(camera_ray_bundle, RayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))

    # Test generate rays on 1D input
    num_rays = 10
    intrinsics = torch.ones(num_rays, 4)  # (num_rays, num_intrinsics_params)
    camera_to_world = torch.ones(num_rays, 3, 4)  # (num_rays, 3, 4)
    coords = torch.ones(num_rays, 2)
    pinhole_camera.generate_rays(intrinsics, camera_to_world, coords)


def test_equirectangular_camera():
    """Test that the equirectangular camera model works."""
    equirectangular_camera = EquirectangularCamera(height=400, width=400)
    camera_ray_bundle = equirectangular_camera.get_camera_ray_bundle()
    assert isinstance(camera_ray_bundle, RayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))

    # Test generate rays on 1D input
    num_rays = 10
    intrinsics = torch.ones(num_rays, 2) * 20  # (num_rays, num_intrinsics_params)
    camera_to_world = torch.ones(num_rays, 3, 4)  # (num_rays, 3, 4)
    coords = torch.ones(num_rays, 2)
    equirectangular_camera.generate_rays(intrinsics, camera_to_world, coords)


if __name__ == "__main__":
    test_pinhole_camera()
    test_equirectangular_camera()
