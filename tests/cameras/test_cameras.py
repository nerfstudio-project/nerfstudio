"""
Test the camera classes.
"""
import dataclasses
from itertools import product

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
    h_w = 800
    fx_y = 10.0
    cx_y = h_w / 2.0
    # pylint: disable=unnecessary-comprehension
    c2w_flat = torch.eye(4)[:3, :]
    camera_to_worlds = [
        c2w_flat,
        torch.stack([c2w_flat] * batch_size),
        torch.stack([torch.stack([c2w_flat] * batch_size)] * batch_size),
    ]
    fx_ys = [
        fx_y,
        torch.tensor(1).float() * fx_y,
        torch.ones(batch_size) * fx_y,
        torch.ones(batch_size, 1) * fx_y,
        torch.ones((batch_size, batch_size)) * fx_y,
    ]
    h_ws = [
        None,
        h_w,
        torch.tensor(1) * h_w,
        torch.ones(batch_size).int() * h_w,
        torch.ones(batch_size, 1).int() * h_w,
        torch.ones((batch_size, batch_size)).int() * h_w,
    ]
    cx_ys = [
        cx_y,
        torch.tensor(1).float() * cx_y,
        torch.ones(batch_size) * cx_y,
        torch.ones(batch_size, 1) * cx_y,
        torch.ones((batch_size, batch_size)) * cx_y,
    ]
    distortion_params = [None, torch.zeros(6), torch.zeros((batch_size, 6)), torch.zeros((batch_size, batch_size, 6))]
    camera_types = [
        1,
        torch.tensor(1),
        torch.ones(batch_size).int(),
        torch.ones(batch_size, 1).int(),
        torch.ones((batch_size, batch_size)).int(),
    ]

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

    c0 = Cameras(
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

    c1 = Cameras(
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

    c2 = Cameras(
        camera_to_worlds[2],
        fx_ys[0],
        fx_ys[0],
        cx_ys[0],
        cx_ys[0],
        h_ws[0],
        h_ws[0],
        distortion_params[0],
        camera_types[0],
    )

    c2_dist = Cameras(
        camera_to_worlds[2],
        fx_ys[0],
        fx_ys[0],
        cx_ys[0],
        cx_ys[0],
        h_ws[0],
        h_ws[0],
        distortion_params[1],
        camera_types[0],
    )

    _ = c2_dist[torch.tensor([0]), torch.tensor([0])]

    assert c0.shape == ()
    assert c1[...].shape == torch.Size([batch_size])

    assert _check_cam_shapes(c0, ())
    assert _check_cam_shapes(c1, (2,))
    assert _check_cam_shapes(c2_dist, (2, 2))

    assert c0.generate_rays(0).shape == (h_w, h_w)
    assert c0.generate_rays(0, coords=torch.ones(10, 2)).shape == (10,)
    c1.generate_rays(0)
    c1.generate_rays(torch.tensor([0, 1]).unsqueeze(-1))

    # Make sure rays generated are same when distortion params are identity (all zeros) and None
    assert c2.shape == c2_dist.shape

    c2_rays = c2.generate_rays(torch.tensor([0, 0]))
    c_dist_rays = c2_dist.generate_rays(torch.tensor([0, 0]))
    assert _check_dataclass_allclose(c2_rays, c_dist_rays)
    assert c2_rays.shape == (h_w, h_w)
    assert c_dist_rays.shape == (h_w, h_w)

    camera_indices = torch.tensor([[0, 0]])
    assert c2.generate_rays(camera_indices).shape == (h_w, h_w, 1)

    for args in product(
        camera_to_worlds,
        fx_ys,
        fx_ys[:-2],
        cx_ys,
        cx_ys[:-2],
        h_ws,
        h_ws[:-2],
        distortion_params[:-1],
        camera_types[:-1],
    ):
        c = Cameras(*args)
        assert len(c.shape) <= 2

    c = c0
    assert c.shape == ()
    coord = torch.tensor([1, 1])
    combos = [
        (0, None),
        (0, coord),
        (torch.zeros(1, 1), coord.broadcast_to(1, 2)),
        (0, coord.broadcast_to(1, 2)),
        (torch.zeros(1), None),
        (torch.zeros(5, 1), coord.broadcast_to(5, 2)),
        (0, coord.broadcast_to(5, 2)),
        (torch.zeros(5, 1), None),
        (torch.zeros(11, 5, 1), coord.broadcast_to(11, 5, 2)),
        (0, coord.broadcast_to(11, 5, 2)),
        (torch.zeros(11, 5, 1), None),
    ]
    for camera_indices, coords in combos:
        c.generate_rays(camera_indices=camera_indices, coords=coords)

    c = c1
    assert c.shape == (2,)
    for camera_indices, coords in combos:
        c.generate_rays(camera_indices=camera_indices, coords=coords)

    c = c2
    combos = [
        (torch.zeros(1, 2), coord.broadcast_to(1, 2)),
        (torch.zeros(2), None),
        (torch.zeros(5, 2), coord.broadcast_to(5, 2)),
        (torch.zeros(5, 2), None),
        (torch.zeros(11, 5, 2), coord.broadcast_to(11, 5, 2)),
        (torch.zeros(11, 5, 2), None),
    ]
    for camera_indices, coords in combos:
        c.generate_rays(camera_indices=camera_indices, coords=coords)


def _check_dataclass_allclose(ipt, other):
    for field in dataclasses.fields(ipt):
        if getattr(ipt, field.name) is not None:
            assert torch.allclose(getattr(ipt, field.name), getattr(other, field.name))
    return True


def _check_cam_shapes(c: Cameras, batch_size):
    if batch_size:
        assert len(c) == batch_size[0]
    assert c.shape == (*batch_size,)
    assert c.camera_to_worlds.shape == (*batch_size, 3, 4)
    assert c.fx.shape == (*batch_size, 1)
    assert c.fy.shape == (*batch_size, 1)
    assert c.cx.shape == (*batch_size, 1)
    assert c.cy.shape == (*batch_size, 1)
    assert c.height.shape == (*batch_size, 1)
    assert c.width.shape == (*batch_size, 1)
    assert c.distortion_params is None or c.distortion_params.shape == (*batch_size, 6)
    assert c.camera_type.shape == (*batch_size, 1)
    return True


if __name__ == "__main__":
    test_pinhole_camera()
    test_camera_as_tensordataclass()
