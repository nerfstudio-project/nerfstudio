import torch

from mattport.structures.cameras import PinholeCamera
from mattport.structures.rays import RayBundle, CameraRayBundle


def test_pinhole_camera():
    pinhole_camera = PinholeCamera(cx=400.0, cy=400.0, fx=10.0, fy=10.0)
    camera_ray_bundle = pinhole_camera.generate_camera_rays()
    assert isinstance(camera_ray_bundle, CameraRayBundle)
    assert torch.allclose(camera_ray_bundle.origins[0], torch.tensor([0.0, 0.0, 0.0]))
    # assert False
