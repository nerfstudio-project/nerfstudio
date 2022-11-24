"""
Test the camera classes.
"""

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle


def test_pinhole_camera():
    """Test that the pinhole camera model works."""
    c2w = torch.eye(4)[None, :3, :]
    pinhole_camera = Cameras(cx=400.0, cy=400.0, fx=10.0, fy=10.0, camera_to_worlds=c2w)
    camera_ray_bundle = pinhole_camera.generate_rays(camera_indices=0)
    assert isinstance(camera_ray_bundle, RayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))

    # Test generate rays on 1D input
    num_rays = 10
    coords = torch.ones(num_rays, 2)
    pinhole_camera.generate_rays(camera_indices=0, coords=coords)


def test_equirectangular_camera():
    """Test that the equirectangular camera model works."""
    height = 100  # width is twice the height
    c2w = torch.eye(4)[None, :3, :]
    equirectangular_camera = Cameras(
        cx=float(height),
        cy=0.5 * float(height),
        fx=float(height),
        fy=float(height),
        camera_to_worlds=c2w,
        camera_type=CameraType.EQUIRECTANGULAR,
    )
    camera_ray_bundle = equirectangular_camera.generate_rays(camera_indices=0)
    assert isinstance(camera_ray_bundle, RayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))

    # Check that the directions are mostly correct in local camera coordinates (+y is up, -z is forward)
    directions = camera_ray_bundle.directions
    threshold = 0.9
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    z = torch.tensor([0.0, 0.0, 1.0])
    # top pixels point up
    assert directions[0, 0] @ y > threshold
    assert directions[0, height] @ y > threshold
    assert directions[0, -1] @ y > threshold
    # middle pixels point horizontally; middle of image is camera forwards
    assert directions[height // 2, 0] @ z > threshold
    assert directions[height // 2, height // 2] @ -x > threshold
    assert directions[height // 2, height] @ -z > threshold
    assert directions[height // 2, 3 * height // 2] @ x > threshold
    assert directions[height // 2, -1] @ z > threshold
    # bottom pixels point down
    assert directions[-1, 0] @ -y > threshold
    assert directions[-1, height] @ -y > threshold
    assert directions[-1, -1] @ -y > threshold


if __name__ == "__main__":
    test_pinhole_camera()
