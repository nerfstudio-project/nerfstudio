"""
Test AABB intersection
"""
import importlib
import math
import os
import random

import torch

from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.misc import strtobool

test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _swap_minmax(x, y):
    if x < y:
        return x, y

    return y, x


def _get_random_aabb_box(max_box, device):
    """
    generate random aabb box
    :param max_box:
    :param device:
    :return:
    """
    x_min = random.uniform(-max_box, max_box)
    x_max = random.uniform(-max_box, max_box)

    x_min, x_max = _swap_minmax(x_min, x_max)

    y_min = random.uniform(-max_box, max_box)
    y_max = random.uniform(-max_box, max_box)

    y_min, y_max = _swap_minmax(y_min, y_max)

    z_min = random.uniform(-max_box, max_box)
    z_max = random.uniform(-max_box, max_box)

    z_min, z_max = _swap_minmax(z_min, z_max)

    return torch.tensor([x_min, y_min, z_min, x_max, y_max, z_max], dtype=torch.float32, device=device)


def _get_random_rays(num_rays, max_box, aabb, device):
    """
    generate random rays that distribute in the general direction of the box.
    The implementation is naive for now.
    :param num_rays:
    :param max_box:
    :param aabb:
    :param device:
    :return:
    """
    origins = []
    directions = []
    aabb_center = torch.tensor(
        [0.5 * (aabb[0] + aabb[3]), 0.5 * (aabb[1] + aabb[4]), 0.5 * (aabb[2] + aabb[5])],
        dtype=torch.float32,
        device=device,
    )

    for _ in range(num_rays):
        x = random.uniform(-max_box, max_box)
        y = random.uniform(-max_box, max_box)
        z = random.uniform(-max_box, max_box)

        origin = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        dirs_to_cent = aabb_center - origin
        dirs_to_cent = dirs_to_cent / torch.linalg.norm(dirs_to_cent)
        phi = math.asin(dirs_to_cent[2])
        theta = torch.atan2(dirs_to_cent[1], dirs_to_cent[0])

        error_ang_phi = random.gauss(0, 0.05)
        error_ang_theta = random.gauss(0, 0.1)

        phi = phi + error_ang_phi
        theta = theta + error_ang_theta

        x_dir = math.cos(phi) * math.cos(theta)
        y_dir = math.cos(phi) * math.sin(theta)
        z_dir = math.sin(phi)

        origins.append([x, y, z])
        directions.append([x_dir, y_dir, z_dir])

    origins = torch.tensor(origins, dtype=torch.float32, device=device)
    directions = torch.tensor(directions, dtype=torch.float32, device=device)

    return origins, directions


def _is_point_on_aabb_boundary(point, aabb):
    # check
    error_threshold = 0.001
    for dim in range(3):
        if point[dim] < (aabb[dim] - error_threshold) or point[dim] > (aabb[dim + 3] + error_threshold):
            return False
    # check that the points is near the boundary
    for side in range(2):
        for dim in range(3):
            if abs(point[dim] - aabb[side * 3 + dim]) < error_threshold:
                return True
    return False


def _assert_points_on_aabb_boundary(points, aabb):
    for point in points:
        assert _is_point_on_aabb_boundary(point, aabb)


def test_rays_intersection_are_on_aabb_boundary():
    """
    test that org + dir*t_min and org + dir*t_max are on aabb boundary
    :return:
    """
    random.seed(496)
    max_box_size = 100
    aabb = _get_random_aabb_box(max_box_size, test_device)
    num_rays = 500
    max_test_size = 1000
    origins, directions = _get_random_rays(num_rays, max_test_size, aabb, test_device)

    t_min, t_max = intersect_aabb(origins, directions, aabb)

    mask = t_min < 1e10
    t_min = t_min[mask]
    t_max = t_max[mask]
    origins = origins[mask]
    directions = directions[mask]

    # assert that we don't do something trivial
    assert len(t_min) > 10

    # assert that tmax is larger than t_min
    assert torch.all(t_max >= t_min)

    positions_min = origins + directions * t_min.view(-1, 1)
    positions_max = origins + directions * t_max.view(-1, 1)

    _assert_points_on_aabb_boundary(positions_min, aabb)
    _assert_points_on_aabb_boundary(positions_max, aabb)


def test_equall_nerfacc():
    """
    test that the output of intersect_aabb is close to nerfacc.intersect_aabb
    :return:
    """
    check_nerf_acc = strtobool(os.environ.get("INTERSECT_WITH_NERFACC", "FALSE"))
    if check_nerf_acc:
        nerfacc = importlib.import_module("nerfacc")
        random.seed(8128)

        max_box_size = 100
        aabb = _get_random_aabb_box(max_box_size, test_device)
        num_rays = 500
        max_test_size = 1000

        origins, directions = _get_random_rays(num_rays, max_test_size, aabb, test_device)

        # time1 = time.time()
        t_min, t_max = intersect_aabb(origins, directions, aabb)
        # time2 = time.time()

        # time3 = time.time()
        t_min_nerfacc, t_max_nerfacc, _ = nerfacc.ray_aabb_intersect(
            origins, directions, aabb[None, :], near_plane=0, far_plane=1e10, miss_value=1e10
        )
        t_max_nerfacc = t_max_nerfacc.squeeze(-1)
        t_min_nerfacc = t_min_nerfacc.squeeze(-1)
        # time4 = time.time()

        # print("pytorch ", time2-time1)
        # print("nerfacc ", time4-time3)

        assert torch.allclose(t_min, t_min_nerfacc, rtol=0.001)
        assert torch.allclose(t_max, t_max_nerfacc, rtol=0.001)
