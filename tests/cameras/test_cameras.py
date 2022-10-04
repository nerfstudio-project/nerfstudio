"""
Test the camera classes.
"""

import torch

from nerfstudio.cameras.cameras import Cameras
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


def test_camera_as_tensordataclass():
    """Test that the camera class move to Tensordataclass works."""
    batch_size = 2
    # pylint: disable=unnecessary-comprehension
    c2w_flat = torch.tensor([i for i in range(12)]).reshape((3, 4))
    camera_to_worlds = [
        c2w_flat,
        torch.stack([c2w_flat] * batch_size),
        torch.stack([torch.stack([c2w_flat] * batch_size)] * batch_size),
    ]
    fx_ys = [1.0, torch.tensor(1).float(), torch.ones(batch_size), torch.ones((batch_size, batch_size))]
    h_ws = [1, torch.tensor(1), torch.ones(batch_size).int(), torch.ones((batch_size, batch_size)).int()]
    cx_ys = fx_ys
    distortion_params = [None, torch.zeros(6), torch.zeros((batch_size, 6)), torch.zeros((batch_size, batch_size, 6))]
    camera_types = [1, torch.tensor(1), torch.ones(batch_size).int(), torch.ones((batch_size, batch_size)).int()]

    c = Cameras(
        camera_to_worlds[1],
        fx_ys[0],
        fx_ys[0],
        cx_ys[0],
        cx_ys[0],
        h_ws[0],
        h_ws[0],
        distortion_params[0],
        camera_types[0],
    )

    # pylint: disable=too-many-nested-blocks
    for camera_to_world in camera_to_worlds:
        for fx in fx_ys:
            for fy in fx_ys:
                for cx in cx_ys:
                    for cy in cx_ys:
                        for distortion_param in distortion_params:
                            for h in h_ws:
                                for w in h_ws:
                                    for camera_type in reversed(camera_types):
                                        c = Cameras(
                                            camera_to_world, fx, fy, cx, cy, w, h, distortion_param, camera_type
                                        )
                                        assert len(c.shape) <= 2

    c = Cameras(
        camera_to_worlds[0],
        fx_ys[0],
        fx_ys[0],
        cx_ys[0],
        cx_ys[0],
        h_ws[0],
        h_ws[0],
        distortion_params[0],
        camera_types[0],
    )
    assert c.shape == ()
    assert Cameras(
        camera_to_worlds[1],
        fx_ys[0],
        fx_ys[0],
        cx_ys[0],
        cx_ys[0],
        h_ws[0],
        h_ws[0],
        distortion_params[0],
        camera_types[0],
    )[...].shape == torch.Size([batch_size])


if __name__ == "__main__":
    test_pinhole_camera()
    test_camera_as_tensordataclass()
