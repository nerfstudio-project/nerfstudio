# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Skipped singular Newton steps must not contaminate the backward pass."""

import pytest
import torch

from nerfstudio.cameras.camera_utils import radial_and_tangential_undistort


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_singular_undistortion_skips_step_with_finite_gradient(iterations, dtype):
    coords = torch.tensor([[1.0, 0.0]], dtype=dtype, requires_grad=True)
    params = torch.tensor([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)
    output = radial_and_tangential_undistort(coords, params, max_iterations=iterations)
    torch.testing.assert_close(output, coords)
    coords_grad, params_grad = torch.autograd.grad(output.sum(), (coords, params))
    torch.testing.assert_close(coords_grad, torch.ones_like(coords))
    torch.testing.assert_close(params_grad, torch.zeros_like(params))


def test_mixed_singular_and_regular_points_keep_regular_newton_step():
    coords = torch.tensor([[1.0, 0.0], [0.2, 0.1]], dtype=torch.float64, requires_grad=True)
    params = torch.tensor(
        [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    output = radial_and_tangential_undistort(coords, params, max_iterations=1)
    x = coords[1]
    k = params[1, 0]
    residual = x * k * x.square().sum()
    J = (1 + k * x.square().sum()) * torch.eye(2, dtype=x.dtype) + 2 * k * x[:, None] * x[None, :]
    expected = torch.stack([coords[0], x - torch.linalg.solve(J, residual)])
    torch.testing.assert_close(output, expected)
    actual_grad = torch.autograd.grad(output.sum(), (coords, params), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (coords, params))
    assert all(torch.isfinite(gradient).all() for gradient in actual_grad)
    torch.testing.assert_close(actual_grad[0], expected_grad[0])
    torch.testing.assert_close(actual_grad[1][:, 0], expected_grad[1][:, 0])
    torch.testing.assert_close(actual_grad[1][0], torch.zeros_like(params[0]))
