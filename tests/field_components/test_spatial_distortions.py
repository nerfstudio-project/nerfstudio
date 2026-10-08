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

"""Tests for scene contraction values and derivatives."""

import pytest
import torch

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.utils.math import Gaussians


@pytest.mark.parametrize("order", [None, 2, float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scene_contraction_identity_inside(order, dtype):
    """The contraction is locally the identity throughout the unit ball."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.3, 0.1]], dtype=dtype, requires_grad=True)
    contracted = SceneContraction(order)(positions)
    torch.testing.assert_close(contracted, positions)

    gradient = torch.autograd.grad(contracted.square().sum(), positions, create_graph=True)[0]
    torch.testing.assert_close(gradient, 2 * positions)
    hessian_sum = torch.autograd.grad(gradient.sum(), positions)[0]
    torch.testing.assert_close(hessian_sum, torch.full_like(positions, 2))


@pytest.mark.parametrize("order", [None, 2, float("inf")])
def test_scene_contraction_gradchecks(order):
    """Autograd agrees with numerical derivatives in mixed inner/outer batches."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.3, 0.1], [2.0, 0.4, -0.2]], dtype=torch.float64, requires_grad=True
    )
    contraction = SceneContraction(order)
    assert torch.autograd.gradcheck(contraction, (positions,))
    assert torch.autograd.gradgradcheck(contraction, (positions,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scene_contraction_gaussians(dtype):
    """Gaussian contraction preserves inner covariance and the radial Jacobian."""
    means = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.3, 0.1], [2.0, 0.4, -0.2]], dtype=dtype, requires_grad=True)
    covariances = torch.diag(torch.tensor([0.1, 0.2, 0.3], dtype=dtype)).expand(3, 3, 3).clone().requires_grad_()
    contracted = SceneContraction()(Gaussians(mean=means, cov=covariances))

    outer_mean = means[-1]
    radius = outer_mean.norm()
    radial_scale = 2 / radius - 1 / radius.square()
    jacobian = radial_scale * torch.eye(3, dtype=dtype)
    jacobian = jacobian + 2 * (1 - radius) / radius**4 * torch.outer(outer_mean, outer_mean)
    expected_means = torch.cat([means[:2], (radial_scale * outer_mean)[None]])
    expected_covariances = torch.cat([covariances[:2], (jacobian @ covariances[-1] @ jacobian.T)[None]])
    torch.testing.assert_close(contracted.mean, expected_means)
    torch.testing.assert_close(contracted.cov, expected_covariances)

    actual_gradients = torch.autograd.grad(contracted.mean.sum() + contracted.cov.sum(), (means, covariances))
    expected_gradients = torch.autograd.grad(expected_means.sum() + expected_covariances.sum(), (means, covariances))
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
