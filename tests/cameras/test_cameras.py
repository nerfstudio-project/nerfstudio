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
    # pylint: disable=unnecessary-comprehension
    c2w_flat = torch.eye(4)[:3, :]
    camera_to_worlds = [
        c2w_flat,
        torch.stack([c2w_flat] * batch_size),
        torch.stack([torch.stack([c2w_flat] * batch_size)] * batch_size),
    ]
    fx_ys = [
        10.0,
        torch.tensor(1).float() * 10,
        torch.ones(batch_size) * 10,
        torch.ones(batch_size, 1) * 10,
        torch.ones((batch_size, batch_size)) * 10,
    ]
    h_ws = [
        None,
        800,
        torch.tensor(1) * 800,
        torch.ones(batch_size).int() * 800,
        torch.ones(batch_size, 1).int() * 800,
        torch.ones((batch_size, batch_size)).int() * 800,
    ]
    cx_ys = [
        400.0,
        torch.tensor(1).float() * 400,
        torch.ones(batch_size) * 400,
        torch.ones(batch_size, 1) * 400,
        torch.ones((batch_size, batch_size)) * 400,
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

    c1_dist = Cameras(
        camera_to_worlds[1],
        fx_ys[0],
        fx_ys[0],
        cx_ys[0],
        cx_ys[0],
        h_ws[0],
        h_ws[0],
        distortion_params[1],
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

    assert c0.shape == ()
    assert c1[...].shape == torch.Size([batch_size])

    assert _check_cam_shapes(c0, ())
    assert _check_cam_shapes(c1, (2,))
    assert _check_cam_shapes(c2_dist, (2, 2))

    c0.generate_rays(0)
    c1.generate_rays(0)

    # Make sure rays generated are same when distortion params are identity (all zeros) and None
    assert c2.shape == c2_dist.shape
    c2_rays = c2.generate_rays(0)
    c_dist_rays = c2_dist.generate_rays(0)
    assert _check_dataclass_allclose(c2_rays, c_dist_rays)

    # assert tensor_dataclass.size == 24
    # assert tensor_dataclass.ndim == 2
    # assert len(tensor_dataclass) == 4

    # reshaped = tensor_dataclass.reshape((2, 12))
    # assert reshaped.shape == (2, 12)
    # assert reshaped.a.shape == (2, 12, 3)
    # assert reshaped.b.shape == (2, 12, 2)
    # assert reshaped.d["t1"].shape == (2, 12, 3)
    # assert reshaped.d["t2"]["t3"].shape == (2, 12, 4)

    # flattened = tensor_dataclass.flatten()
    # assert flattened.shape == (24,)
    # assert flattened.a.shape == (24, 3)
    # assert flattened.b.shape == (24, 2)
    # assert flattened.d["t1"].shape == (24, 3)
    # assert flattened.d["t2"]["t3"].shape == (24, 4)
    # assert flattened[0:4].shape == (4,)

    # # Test indexing operations
    # assert tensor_dataclass[:, 1].shape == (4,)
    # assert tensor_dataclass[:, 1].a.shape == (4, 3)
    # assert tensor_dataclass[:, 1].d["t1"].shape == (4, 3)
    # assert tensor_dataclass[:, 1].d["t2"]["t3"].shape == (4, 4)
    # assert tensor_dataclass[:, 0:2].shape == (4, 2)
    # assert tensor_dataclass[:, 0:2].a.shape == (4, 2, 3)
    # assert tensor_dataclass[:, 0:2].d["t1"].shape == (4, 2, 3)
    # assert tensor_dataclass[:, 0:2].d["t2"]["t3"].shape == (4, 2, 4)
    # assert tensor_dataclass[..., 1].shape == (4,)
    # assert tensor_dataclass[..., 1].a.shape == (4, 3)
    # assert tensor_dataclass[0].shape == (6,)
    # assert tensor_dataclass[0].a.shape == (6, 3)
    # assert tensor_dataclass[0].d["t1"].shape == (6, 3)
    # assert tensor_dataclass[0].d["t2"]["t3"].shape == (6, 4)
    # assert tensor_dataclass[0, ...].shape == (6,)
    # assert tensor_dataclass[0, ...].a.shape == (6, 3)

    # tensor_dataclass = DummyTensorDataclass(
    #     a=torch.ones((2, 3, 4, 5)),
    #     b=torch.ones((4, 5)),
    #     d={"t1": torch.ones((2, 3, 4, 5)), "t2": {"t3": torch.ones((4, 5))}},
    # )
    # assert tensor_dataclass[0, ...].shape == (3, 4)
    # assert tensor_dataclass[0, ...].a.shape == (3, 4, 5)
    # assert tensor_dataclass[0, ...].d["t1"].shape == (3, 4, 5)
    # assert tensor_dataclass[0, ...].d["t2"]["t3"].shape == (3, 4, 5)
    # assert tensor_dataclass[0, ..., 0].shape == (3,)
    # assert tensor_dataclass[0, ..., 0].a.shape == (3, 5)
    # assert tensor_dataclass[0, ..., 0].d["t1"].shape == (3, 5)
    # assert tensor_dataclass[0, ..., 0].d["t2"]["t3"].shape == (3, 5)
    # assert tensor_dataclass[..., 0].shape == (2, 3)
    # assert tensor_dataclass[..., 0].a.shape == (2, 3, 5)
    # assert tensor_dataclass[..., 0].d["t1"].shape == (2, 3, 5)
    # assert tensor_dataclass[..., 0].d["t2"]["t3"].shape == (2, 3, 5)

    # mask = torch.rand(size=(2, 3)) > 0.5
    # assert tensor_dataclass[mask].ndim == 2

    # Making sure cameras don't error on every combination of inputs
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


def _check_dataclass_allclose(input, other):
    for field in dataclasses.fields(input):
        if getattr(input, field.name) is not None:
            assert torch.allclose(getattr(input, field.name), getattr(other, field.name))
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
