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

"""Volume compositing should preserve all leading ray batch dimensions."""

import pytest
import torch

from nerfstudio.cameras.rays import Frustums, RaySamples


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weight_helpers_preserve_ray_batch_shape(batch_shape, dtype):
    shape = batch_shape + (4, 1)
    tau = torch.linspace(0.1, 0.7, 4, dtype=dtype).reshape((1,) * len(batch_shape) + (4, 1)).expand(shape)
    starts = torch.zeros(shape, dtype=dtype)
    samples = RaySamples(
        frustums=Frustums(
            origins=torch.zeros(batch_shape + (4, 3), dtype=dtype),
            directions=torch.ones(batch_shape + (4, 3), dtype=dtype),
            starts=starts,
            ends=starts + 1,
            pixel_area=starts + 1,
        ),
        deltas=torch.ones_like(tau),
    )
    alphas = -torch.expm1(-tau)
    expected_transmittance = torch.exp(-tau.cumsum(-2) + tau)
    torch.testing.assert_close(samples.get_weights(tau), alphas * expected_transmittance)
    weights, transmittance = RaySamples.get_weights_and_transmittance_from_alphas(alphas)
    reference_transmittance = torch.cat(
        [alphas.new_ones(batch_shape + (1, 1)), (1 - alphas + 1e-7).cumprod(-2)], dim=-2
    )
    assert transmittance.dtype == dtype
    torch.testing.assert_close(transmittance, reference_transmittance)
    torch.testing.assert_close(weights, alphas * reference_transmittance[..., :-1, :])
    torch.testing.assert_close(RaySamples.get_weights_and_transmittance_from_alphas(alphas, weights_only=True), weights)
