# Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test exact slab geometry when ray directions contain zero components."""

import pytest
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.utils.math import intersect_aabb, intersect_obb


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("negative_zero", [False, True])
def test_parallel_rays_on_faces_and_edges_have_finite_intervals(dtype, negative_zero):
    aabb = torch.tensor([0, 0, 0, 1, 1, 1], dtype=dtype)
    origins = torch.tensor([[0, 0.5, -1], [1, 0.5, -1], [0, 0, -1], [1, 1, -1]], dtype=dtype)
    zero = -0.0 if negative_zero else 0.0
    directions = torch.tensor([[zero, zero, 1]], dtype=dtype).expand_as(origins)
    near, far = intersect_aabb(origins, directions, aabb)
    torch.testing.assert_close(near, torch.ones_like(near), rtol=0, atol=0)
    torch.testing.assert_close(far, torch.full_like(far, 2), rtol=0, atol=0)


def test_parallel_rays_outside_slabs_are_misses():
    aabb = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float64)
    origins = torch.tensor([[-0.5, 0.5, -1], [1.5, 0.5, -1], [0.5, -0.1, -1]], dtype=torch.float64)
    directions = origins.new_tensor([[0, 0, 1]]).expand_as(origins)
    near, far = intersect_aabb(origins, directions, aabb, max_bound=11, invalid_value=123)
    torch.testing.assert_close(near, torch.full_like(near, 123), rtol=0, atol=0)
    torch.testing.assert_close(far, torch.full_like(far, 123), rtol=0, atol=0)


def test_ray_starting_on_face_inside_box_has_zero_near():
    origins = torch.tensor([[0, 1, 0.5]], dtype=torch.float64)
    directions = origins.new_tensor([[0, 0, 1]])
    aabb = origins.new_tensor([0, 0, 0, 1, 1, 1])
    near, far = intersect_aabb(origins, directions, aabb)
    torch.testing.assert_close(near, near.new_tensor([0]), rtol=0, atol=0)
    torch.testing.assert_close(far, far.new_tensor([0.5]), rtol=0, atol=0)


def test_interior_parallel_intersection_has_finite_analytic_gradients():
    origins = torch.tensor([[0.5, 0.5, -1]], dtype=torch.float64, requires_grad=True)
    directions = torch.tensor([[0, 0, 1]], dtype=torch.float64, requires_grad=True)
    aabb = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float64, requires_grad=True)
    near, far = intersect_aabb(origins, directions, aabb)
    origin_grad, direction_grad, box_grad = torch.autograd.grad((near + far).sum(), (origins, directions, aabb))
    torch.testing.assert_close(origin_grad, origins.new_tensor([[0, 0, -2]]))
    torch.testing.assert_close(direction_grad, directions.new_tensor([[0, 0, -3]]))
    torch.testing.assert_close(box_grad, aabb.new_tensor([0, 0, 1, 0, 0, 1]))
    assert torch.autograd.gradcheck(lambda o, d, b: intersect_aabb(o, d, b), (origins, directions, aabb))


def test_parallel_misses_have_finite_zero_gradients():
    origins = torch.tensor([[-0.5, 0.5, -1]], dtype=torch.float64, requires_grad=True)
    directions = torch.tensor([[0, 0, 1]], dtype=torch.float64, requires_grad=True)
    aabb = origins.new_tensor([0, 0, 0, 1, 1, 1])
    near, far = intersect_aabb(origins, directions, aabb)
    for gradient in torch.autograd.grad((near + far).sum(), (origins, directions)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)


def test_nonzero_small_directions_are_not_treated_as_parallel():
    origins = torch.tensor([[-1e-12, 0.5, -1]], dtype=torch.float64)
    directions = origins.new_tensor([[1e-12, 0, 1]])
    aabb = origins.new_tensor([0, 0, 0, 1, 1, 1])
    near, far = intersect_aabb(origins, directions, aabb)
    torch.testing.assert_close(near, near.new_tensor([1]), rtol=0, atol=0)
    torch.testing.assert_close(far, far.new_tensor([2]), rtol=0, atol=0)


def test_oriented_crop_box_handles_ray_on_parallel_face():
    rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    translation = rotation.new_tensor([2, 3, 4])
    obb = OrientedBox(R=rotation, T=translation, S=rotation.new_tensor([2, 2, 2]))
    origins = rotation.new_tensor([[1, 0, -2]]) @ rotation.T + translation
    directions = rotation.new_tensor([[0, 0, 1]]) @ rotation.T
    near, far = intersect_obb(origins, directions, obb)
    torch.testing.assert_close(near, near.new_tensor([1]), rtol=0, atol=0)
    torch.testing.assert_close(far, far.new_tensor([3]), rtol=0, atol=0)


def test_camera_center_ray_has_finite_crop_bounds_on_box_face():
    camera_to_world = torch.eye(4)[:3].clone()
    camera_to_world[:, 3] = torch.tensor([0, 0.5, 2])
    camera = Cameras(camera_to_worlds=camera_to_world, fx=100.0, fy=100.0, cx=0.5, cy=0.5, height=1, width=1)
    crop = SceneBox(aabb=torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32))
    rays = camera.generate_rays(0, aabb_box=crop)
    torch.testing.assert_close(rays.directions, rays.directions.new_tensor([[[0, 0, -1]]]), rtol=0, atol=0)
    assert rays.nears is not None and rays.fars is not None
    torch.testing.assert_close(rays.nears, rays.nears.new_tensor([[[1]]]), rtol=0, atol=0)
    torch.testing.assert_close(rays.fars, rays.fars.new_tensor([[[2]]]), rtol=0, atol=0)


def test_nonparallel_rays_match_original_formula_and_gradients():
    generator = torch.Generator().manual_seed(19)
    origins = torch.randn(12, 3, generator=generator, dtype=torch.float64).requires_grad_()
    directions = torch.randn(12, 3, generator=generator, dtype=torch.float64).requires_grad_()
    aabb = origins.new_tensor([-2, -2, -2, 2, 2, 2])
    slabs = torch.stack(((aabb[:3] - origins) / directions, (aabb[3:] - origins) / directions))
    near = slabs.amin(0).amax(-1).clamp(0, 1e10)
    far = slabs.amax(0).amin(-1).clamp(0, 1e10)
    miss = far <= near
    expected = (torch.where(miss, 1e10, near), torch.where(miss, 1e10, far))
    actual = intersect_aabb(origins, directions, aabb)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(sum(t.sum() for t in actual), (origins, directions))
    expected_grads = torch.autograd.grad(sum(t.sum() for t in expected), (origins, directions))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_empty_ray_batch_preserves_shapes():
    origins = torch.empty(0, 3)
    aabb = origins.new_tensor([0, 0, 0, 1, 1, 1])
    near, far = intersect_aabb(origins, origins, aabb)
    assert near.shape == far.shape == (0,)
