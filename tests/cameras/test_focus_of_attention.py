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

"""Test least-squares camera focus with identifiable and parallel optical axes."""

import pytest
import torch

from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses, focus_of_attention


def _poses(origins, directions):
    directions = torch.nn.functional.normalize(directions, dim=-1)
    backward = -directions
    reference = torch.zeros_like(backward)
    reference[:, 0] = 1
    reference[backward[:, 0].abs() > 0.9] = reference.new_tensor([0, 1, 0])
    right = torch.nn.functional.normalize(torch.cross(reference, backward, dim=-1), dim=-1)
    up = torch.cross(backward, right, dim=-1)
    poses = torch.eye(4, dtype=origins.dtype, device=origins.device).repeat(len(origins), 1, 1)
    poses[:, :3, :3] = torch.stack((right, up, backward), dim=-1)
    poses[:, :3, 3] = origins
    return poses


def _parallel_poses(dtype=torch.float64):
    origins = torch.tensor([[-2, 1, 0], [0, -1, 1], [1, 1, 2], [3, 5, 3], [5, 7, 4]], dtype=dtype)
    return _poses(origins, origins.new_tensor([0, 0, -1]).expand_as(origins))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parallel_focus_preserves_unidentifiable_depth(dtype):
    poses = _parallel_poses(dtype)
    initial = poses[:, :3, 3].mean(0)
    # Only the cameras at z3/z4 initially see the focus. The nearest-ray
    # least-squares problem identifies x/y, while z must retain the prior.
    expected = initial.new_tensor([4, 6, 2])
    torch.testing.assert_close(focus_of_attention(poses, initial), expected)


def test_parallel_solution_is_nearest_minimizer_to_initial_focus():
    poses = _parallel_poses()
    initial = poses.new_tensor([8, -7, -1])
    actual = focus_of_attention(poses, initial)
    expected = torch.cat((poses[:, :2, 3].mean(0), initial[2:]))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # Moving the prior along the optical-axis nullspace must move the focus
    # by exactly that amount, rather than reset it to the world origin.
    shift = initial.new_tensor([0, 0, -3])
    shifted = focus_of_attention(poses, initial + shift)
    torch.testing.assert_close(shifted, actual + shift, rtol=1e-12, atol=1e-12)


def test_parallel_auto_centering_does_not_abort_dataset_preparation():
    poses = _parallel_poses(torch.float32)
    centered, transform = auto_orient_and_center_poses(poses, method="none", center_method="focus")
    expected_focus = poses.new_tensor([4, 6, 2])
    torch.testing.assert_close(transform[:3, 3], -expected_focus)
    torch.testing.assert_close(centered[:, :3, 3], poses[:, :3, 3] - expected_focus)
    torch.testing.assert_close(centered[:, :3, :3], poses[:, :3, :3])


def test_focus_is_equivariant_under_rigid_world_transform():
    poses = _parallel_poses()
    initial = poses.new_tensor([8, -7, -1])
    angle = poses.new_tensor(0.7)
    rotation = torch.stack(
        (
            torch.stack((angle.cos(), angle.new_zeros(()), angle.sin())),
            angle.new_tensor([0, 1, 0]),
            torch.stack((-angle.sin(), angle.new_zeros(()), angle.cos())),
        )
    )
    translation = poses.new_tensor([12, -4, 9])
    transformed = poses.clone()
    transformed[:, :3, :3] = rotation @ poses[:, :3, :3]
    transformed[:, :3, 3] = poses[:, :3, 3] @ rotation.T + translation
    actual = focus_of_attention(transformed, rotation @ initial + translation)
    expected = rotation @ focus_of_attention(poses, initial) + translation
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_full_rank_matches_existing_normal_equation_solution():
    origins = torch.tensor([[2, 0, 0], [0, 3, 0], [0, 0, 4], [-2, -3, 2]], dtype=torch.float64)
    target = origins.new_tensor([0.3, -0.2, 0.5])
    offsets = origins.new_tensor([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0.05, 0, 0.03]])
    directions = torch.nn.functional.normalize(target - origins + offsets, dim=-1)
    poses = _poses(origins, directions)
    projection = torch.eye(3, dtype=origins.dtype)[None] - directions[:, :, None] * directions[:, None, :]
    normal = projection.transpose(-1, -2) @ projection
    expected = torch.linalg.solve(normal.mean(0), (normal @ origins[..., None]).mean(0).squeeze(-1))
    actual = focus_of_attention(poses, target)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("num_active", [0, 1])
def test_fewer_than_two_active_cameras_keep_initial_focus(num_active):
    poses = _parallel_poses()
    initial = poses.new_tensor([8, -7, 4 if num_active == 0 else 3.5])
    torch.testing.assert_close(focus_of_attention(poses, initial), initial, rtol=0, atol=0)


def test_nearly_parallel_focus_is_finite_and_preserves_scale():
    poses = _parallel_poses()
    directions = -poses[:, :3, 2]
    directions[:, 0] = torch.arange(len(poses), dtype=poses.dtype) * 1e-10
    poses = _poses(poses[:, :3, 3], directions)
    initial = poses.new_tensor([8, -7, -1])
    actual = focus_of_attention(poses, initial)
    assert torch.isfinite(actual).all()
    assert actual.norm() < 10
    scaled = poses.clone()
    scaled[:, :3, 3] *= 7
    torch.testing.assert_close(focus_of_attention(scaled, initial * 7), actual * 7, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_parallel_focus_keeps_cuda_device():
    poses = _parallel_poses().cuda()
    actual = focus_of_attention(poses, poses[:, :3, 3].mean(0))
    assert actual.device == poses.device
    torch.testing.assert_close(actual, actual.new_tensor([4, 6, 2]))
