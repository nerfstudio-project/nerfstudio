"""
Test the camera classes.
"""
import dataclasses
from itertools import product

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle

BATCH_SIZE = 2
H_W = 800
FX_Y = 10.0
CX_Y = H_W / 2.0
C2W_FLAT = torch.eye(4)[:3, :]
CAMERA_TO_WORLDS = [
    C2W_FLAT,
    torch.stack([C2W_FLAT] * BATCH_SIZE),
    torch.stack([torch.stack([C2W_FLAT] * BATCH_SIZE)] * BATCH_SIZE),
]
FX_YS = [
    FX_Y,
    torch.tensor([1]).float() * FX_Y,
    torch.ones(BATCH_SIZE, 1) * FX_Y,
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)) * FX_Y,
]
H_WS = [
    None,
    H_W,
    torch.tensor([1]) * H_W,
    torch.ones(BATCH_SIZE, 1).int() * H_W,
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)).int() * H_W,
]
CX_YS = [
    CX_Y,
    torch.tensor([1]).float() * CX_Y,
    torch.ones(BATCH_SIZE, 1) * CX_Y,
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)) * CX_Y,
]
DISTORTION_PARAMS = [None, torch.zeros(6), torch.zeros((BATCH_SIZE, 6)), torch.zeros((BATCH_SIZE, BATCH_SIZE, 6))]
camera_types = [
    1,
    torch.tensor([1]),
    torch.ones(BATCH_SIZE, 1).int(),
    torch.ones((BATCH_SIZE, BATCH_SIZE, 1)).int(),
]
C = Cameras(
    CAMERA_TO_WORLDS[1],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C0 = Cameras(
    CAMERA_TO_WORLDS[0],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C1 = Cameras(
    CAMERA_TO_WORLDS[1],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C2 = Cameras(
    CAMERA_TO_WORLDS[2],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[0],
    camera_types[0],
)

C2_DIST = Cameras(
    CAMERA_TO_WORLDS[2],
    FX_YS[0],
    FX_YS[0],
    CX_YS[0],
    CX_YS[0],
    H_WS[0],
    H_WS[0],
    DISTORTION_PARAMS[1],
    camera_types[0],
)


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


def test_orthophoto_camera():
    """Test that the orthographic camera model works."""
    c2w = torch.eye(4)[None, :3, :]
    # apply R and T.
    R = torch.Tensor(
        [
            [0.5, -0.14644661, 0.85355339],
            [0.5, 0.85355339, -0.14644661],
            [-0.70710678, 0.5, 0.5],
        ]
    ).unsqueeze(0)
    T = torch.Tensor([[0.5, 0, -0.5]])
    c2w[..., :3, :3] = R
    c2w[..., :3, 3] = T

    ortho_cam = Cameras(cx=1.5, cy=1.5, fx=1.0, fy=1.0, camera_to_worlds=c2w, camera_type=CameraType.ORTHOPHOTO)
    ortho_rays = ortho_cam.generate_rays(camera_indices=0)
    # campare with `PERSPECTIVE` to validate `ORTHOPHOTO`.
    pinhole_cam = Cameras(cx=1.5, cy=1.5, fx=1.0, fy=1.0, camera_to_worlds=c2w, camera_type=CameraType.PERSPECTIVE)
    pinhole_rays = pinhole_cam.generate_rays(camera_indices=0)

    assert ortho_rays.shape == pinhole_rays.shape
    # `ortho_rays.directions` should equal to the center ray of `pinhole_rays.directions`.
    assert torch.allclose(
        ortho_rays.directions, pinhole_rays.directions[1, 1].broadcast_to(ortho_rays.directions.shape)
    )
    # `ortho_rays.origins` should be grid points with a mean value of `pinhole_rays.origins`.
    assert torch.allclose(ortho_rays.origins.mean(dim=(0, 1)), pinhole_rays.origins[1, 1])


def test_multi_camera_type():
    """Test that the orthographic camera model works."""
    # here we test two different camera types.
    num_cams = [2]
    c2w = torch.eye(4)[None, :3, :].broadcast_to(*num_cams, 3, 4)
    cx = torch.Tensor([20]).broadcast_to(*num_cams, 1)
    cy = torch.Tensor([10]).broadcast_to(*num_cams, 1)
    fx = torch.Tensor([10]).broadcast_to(*num_cams, 1)
    fy = torch.Tensor([10]).broadcast_to(*num_cams, 1)
    h = torch.Tensor([40]).long().broadcast_to(*num_cams, 1)
    w = torch.Tensor([20]).long().broadcast_to(*num_cams, 1)
    camera_type = [CameraType.PERSPECTIVE, CameraType.ORTHOPHOTO]
    multitype_cameras = Cameras(c2w, fx, fy, cx, cy, w, h, camera_type=camera_type)

    # test `generate_rays`, 1 cam.
    ray0 = multitype_cameras.generate_rays(camera_indices=0)
    assert ray0.shape == torch.Size([40, 20])

    # test `generate_rays`, multiple cams.
    num_rays = [30, 30]
    camera_indices = torch.randint(0, 2, [*num_rays, len(num_cams)])  # (*num_rays, num_cameras_batch_dims)
    ray1 = multitype_cameras.generate_rays(camera_indices=camera_indices)
    assert ray1.shape == torch.Size([40, 20, *num_rays])

    # test `_generate_rays_from_coords`, 1 cam.
    coords = torch.randint(0, 10, [*num_rays, 2]).float()  # (*num_rays 2)
    ray2 = multitype_cameras.generate_rays(camera_indices=0, coords=coords)
    assert ray2.shape == torch.Size([*num_rays])

    # test `_generate_rays_from_coords`, multiple cam.
    ray3 = multitype_cameras.generate_rays(camera_indices=camera_indices, coords=coords)
    assert ray3.shape == torch.Size([*num_rays])


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


def test_camera_as_tensordataclass():
    """Test that the camera class move to Tensordataclass works."""
    _ = C2_DIST[torch.tensor([0]), torch.tensor([0])]

    assert C0.shape == ()
    assert C1[...].shape == torch.Size([BATCH_SIZE])

    assert _check_cam_shapes(C0, ())
    assert _check_cam_shapes(C1, (2,))
    assert _check_cam_shapes(C2_DIST, (2, 2))

    assert C0.generate_rays(0).shape == (H_W, H_W)
    assert C0.generate_rays(0, coords=torch.ones(10, 2)).shape == (10,)
    C1.generate_rays(0)
    C1.generate_rays(torch.tensor([0, 1]).unsqueeze(-1))

    # Make sure rays generated are same when distortion params are identity (all zeros) and None
    assert C2.shape == C2_DIST.shape

    c2_rays = C2.generate_rays(torch.tensor([0, 0]))
    c_dist_rays = C2_DIST.generate_rays(torch.tensor([0, 0]))
    assert _check_dataclass_allclose(c2_rays, c_dist_rays)
    assert c2_rays.shape == (H_W, H_W)
    assert c_dist_rays.shape == (H_W, H_W)

    camera_indices = torch.tensor([[0, 0]])
    assert C2.generate_rays(camera_indices).shape == (H_W, H_W, 1)

    for args in product(
        CAMERA_TO_WORLDS,
        FX_YS,
        FX_YS[:1],
        CX_YS,
        CX_YS[:1],
        H_WS,
        H_WS[:1],
        DISTORTION_PARAMS[:-1],
        camera_types[:-1],
    ):
        _ = Cameras(*args)


def check_generate_rays_shape():
    """Checking the output shapes from Cameras.generate_rays"""
    coord = torch.tensor([1, 1])
    combos = [
        (0, None, torch.Size((800, 800))),  # First camera, all pixels
        (
            0,
            coord,
            torch.Size(()),
        ),  # First camera, selected pixels in coords, output zero dimensional since no extra batch dim in coord
        (
            torch.zeros(1, 1),
            coord.broadcast_to(1, 2),
            torch.Size((1,)),
        ),  # [0]th camera and selected coords, output is [1] dimensional
        (
            0,
            coord.broadcast_to(1, 2),
            torch.Size((1,)),
        ),  # First camera and selected coords, output is [1] dimensional since one extra batch dim in coords
        (
            torch.zeros(1),
            None,
            torch.Size((800, 800)),
        ),  # [0]th camera, all pixels, output is [HxW] dimensional since no extra output batch dim
        (
            torch.zeros(5, 1),
            coord.broadcast_to(5, 2),
            torch.Size((5,)),
        ),  # [0]th camera and selected coords, output is [5] dimensional since extra batch dim is provided in inputs
        (0, coord.broadcast_to(5, 2), torch.Size((5,))),  # First camera and selected coords, output is [5] dimensional
        (
            torch.zeros(5, 1),
            None,
            torch.Size((800, 800, 5)),
        ),  # [0]th camera and all pixels, HxWx5 since coords is none and our extra batch dim in our inputs is [5]
        (
            torch.zeros(11, 5, 1),
            coord.broadcast_to(11, 5, 2),
            torch.Size((11, 5)),
        ),  # [0]th camera and selected coords since inputs have 2 extra batch dims of [11,5]
        (
            0,
            coord.broadcast_to(11, 5, 2),
            torch.Size((11, 5)),
        ),  # [0]th camera and selected coords since coords have 2 extra batch dims of [11, 5]
        (
            torch.zeros(11, 5, 1),
            None,
            torch.Size((800, 800, 11, 5)),
        ),  # [0]th camera and all pixels since coords is none but we still have 2 extra batch dims of [11, 5] in coords
    ]
    for camera_indices, coords, output_size in combos:
        shape = C0.generate_rays(camera_indices=camera_indices, coords=coords).shape
        assert shape == output_size

    assert C1.shape == (2,)
    for camera_indices, coords, output_size in combos:
        shape = C1.generate_rays(camera_indices=camera_indices, coords=coords).shape
        assert shape == output_size

    # camera_indices can't be an int anymore since our cameras object have 2 batch dimensions
    # camera_indices last dim needs to be (..., 2) since len(cameras.shape) == 2
    combos = [
        (torch.zeros(2), coord, ()),
        (torch.zeros(1, 2), coord.broadcast_to(1, 2), (1,)),
        (torch.zeros(2), None, (800, 800)),
        (torch.zeros(5, 2), coord.broadcast_to(5, 2), (5,)),
        (torch.zeros(5, 2), None, (800, 800, 5)),
        (torch.zeros(11, 5, 2), coord.broadcast_to(11, 5, 2), (11, 5)),
        (torch.zeros(11, 5, 2), None, (800, 800, 11, 5)),
    ]
    for camera_indices, coords, output_size in combos:
        shape = C2.generate_rays(camera_indices=camera_indices, coords=coords).shape
        assert shape == output_size


def _check_dataclass_allclose(ipt, other):
    for field in dataclasses.fields(ipt):
        if getattr(ipt, field.name) is not None:
            if isinstance(getattr(ipt, field.name), dict):
                ipt_dict = getattr(ipt, field.name)
                other_dict = getattr(other, field.name)
                for k, v in ipt_dict.items():
                    assert k in other_dict
                    assert torch.allclose(v, other_dict[k])
            else:
                assert torch.allclose(getattr(ipt, field.name), getattr(other, field.name))
    return True


def _check_cam_shapes(cam: Cameras, _batch_size):
    if _batch_size:
        assert len(cam) == _batch_size[0]
    assert cam.shape == (*_batch_size,)
    assert cam.camera_to_worlds.shape == (*_batch_size, 3, 4)
    assert cam.fx.shape == (*_batch_size, 1)
    assert cam.fy.shape == (*_batch_size, 1)
    assert cam.cx.shape == (*_batch_size, 1)
    assert cam.cy.shape == (*_batch_size, 1)
    assert cam.height.shape == (*_batch_size, 1)
    assert cam.width.shape == (*_batch_size, 1)
    assert cam.distortion_params is None or cam.distortion_params.shape == (*_batch_size, 6)
    assert cam.camera_type.shape == (*_batch_size, 1)
    return True


if __name__ == "__main__":
    test_pinhole_camera()
    test_equirectangular_camera()
    test_camera_as_tensordataclass()
    test_orthophoto_camera()
    test_multi_camera_type()
