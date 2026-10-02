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

"""Regression tests for differentiable Fisheye624 geometry on the optical axis."""

import pytest
import torch

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper


def _params(dtype, num_params=16):
    params = torch.zeros(1, num_params, dtype=dtype)
    if num_params == 16:
        params[0, :4] = torch.tensor([120.0, 80.0, 31.0, 27.0], dtype=dtype)
    else:
        params[0, :3] = torch.tensor([120.0, 31.0, 27.0], dtype=dtype)
    # Nonzero radial, tangential and thin-prism terms still have identity
    # first derivative in normalized coordinates at the optical axis.
    params[0, -12] = 0.01
    params[0, -6:] = 0.001
    return params


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_params", [15, 16])
def test_project_optical_axis_jacobian(dtype, num_params):
    params = _params(dtype, num_params)
    xyz = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda x: fisheye624_project(x, params), xyz).reshape(2, 3)
    fx, fy = params[0, 0], params[0, 0 if num_params == 15 else 1]
    expected = torch.zeros(2, 3, dtype=dtype)
    expected[0, 0], expected[1, 1] = fx / 2, fy / 2
    torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_params", [15, 16])
def test_unproject_principal_point_jacobian(dtype, num_params):
    params = _params(dtype, num_params)
    center = params[:, 1:3] if num_params == 15 else params[:, 2:4]
    uv = center[:, None].clone().requires_grad_()
    jacobian = torch.autograd.functional.jacobian(lambda x: fisheye624_unproject_helper(x, params), uv).reshape(3, 2)
    fx, fy = params[0, 0], params[0, 0 if num_params == 15 else 1]
    expected = torch.zeros(3, 2, dtype=dtype)
    expected[0, 0], expected[1, 1] = 1 / fx, 1 / fy
    torch.testing.assert_close(jacobian, expected)


def test_project_preserves_direction_near_optical_axis():
    params = torch.zeros(1, 16, dtype=torch.float64)
    params[0, :2] = 1
    xyz = torch.tensor([[[1e-10, 0.0, 1.0], [0.0, -2e-10, 1.0]]], dtype=torch.float64)
    torch.testing.assert_close(fisheye624_project(xyz, params), xyz[..., :2], rtol=1e-12, atol=0)


def test_center_projection_and_unprojection_gradcheck():
    params = _params(torch.float64)
    xyz = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    uv = params[:, None, 2:4].clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x: fisheye624_project(x, params), (xyz,))
    assert torch.autograd.gradcheck(lambda x: fisheye624_unproject_helper(x, params), (uv,))


def test_center_camera_parameter_gradients_are_finite():
    params = _params(torch.float64).requires_grad_()
    xyz = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    projected = fisheye624_project(xyz, params)
    project_grad = torch.autograd.grad(projected.sum(), params)[0]
    expected = torch.zeros_like(params)
    expected[:, 2:4] = 1
    torch.testing.assert_close(project_grad, expected)
    # Treat the image location as fixed when differentiating the intrinsics.
    uv = params[:, None, 2:4].detach().clone()
    rays = fisheye624_unproject_helper(uv, params)
    unproject_grad = torch.autograd.grad(rays.sum(), params)[0]
    expected[:, 2:4] = -1 / params[:, :2].detach()
    torch.testing.assert_close(unproject_grad, expected)


def test_distorted_batched_project_unproject_roundtrip():
    params = _params(torch.float64).repeat(2, 1)
    params[1, :4] = torch.tensor([150.0, 110.0, 42.0, 39.0], dtype=torch.float64)
    xyz = torch.tensor(
        [[[0.0, 0.0, 2.0], [0.3, -0.2, 1.2], [-0.5, 0.8, 2.0]], [[0.0, 0.0, 1.0], [-0.2, -0.1, 0.9], [0.4, 0.2, 1.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = fisheye624_unproject_helper(fisheye624_project(xyz, params), params)
    expected = xyz / xyz[..., 2:3]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual_grad = torch.autograd.grad(actual.sum(), xyz, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), xyz)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)


def test_noncentral_projection_matches_closed_form():
    params = torch.zeros(2, 16, dtype=torch.float64)
    params[:, :2] = torch.tensor([[120.0, 80.0], [150.0, 100.0]])
    params[:, 2:4] = torch.tensor([[31.0, 27.0], [42.0, 39.0]])
    params[:, -12:-6] = torch.tensor([0.01, -0.003, 0.001, -0.0002, 0.0001, -0.00001])
    xyz = torch.tensor([[[0.2, -0.1, 1.0]], [[-0.5, 0.3, 2.0]]], dtype=torch.float64, requires_grad=True)
    ab = xyz[..., :2] / xyz[..., 2:3]
    radius = ab.norm(dim=-1, keepdim=True)
    theta = radius.atan()
    distorted_theta = theta + sum(params[:, None, -12 + i : -11 + i] * theta ** (3 + 2 * i) for i in range(6))
    expected = distorted_theta / radius * ab * params[:, None, :2] + params[:, None, 2:4]
    actual = fisheye624_project(xyz, params)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.sum(), xyz)[0]
    expected_grad = torch.autograd.grad(expected.sum(), xyz)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
