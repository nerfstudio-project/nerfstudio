# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""Collection of Ray Losses"""

import torch
from torchtyping import TensorType

from pyrad.cameras.rays import RaySamples


def distortion_loss(ray_samples: RaySamples, densities: TensorType[..., "num_samples", 1]) -> TensorType[..., 1]:
    """Ray baserd distortion loss proposed in MipNeRF-360.

    .. math::

        \\mathcal{L}(\\mathbf{s}, \\mathbf{w}) =\\iint\\limits_{-\\infty}^{\\,\\,\\,\\infty}
        \\mathbf{w}_\\mathbf{s}(u)\\mathbf{w}_\\mathbf{s}(v)|u - v|\\,d_{u}\\,d_{v}

    where :math:`\\mathbf{w}_\\mathbf{s}(u)=\\sum_i w_i \\mathbb{1}_{[\\mathbf{s}_i, \\mathbf{s}_{i+1})}(u)`
    is the weight at location :math:`u` between bin locations :math:`s_i` and :math:`s_{i+1}`.

    Args:
        ray_samples (RaySamples): Ray samples to compute loss over
        densities (TensorType[..., "num_samples", 1]): Predicted sample densities

    Returns:
        TensorType[..., 1]: Distortion Loss.
    """

    # Compute the weight at each sample location
    weights = ray_samples.get_weights(densities)

    midpoints = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2.0  # (..., num_samples, 1)

    loss = (
        weights * weights[..., None, :, 0] * torch.abs(midpoints - midpoints[..., None, :, 0])
    )  # (..., num_samples, num_samples)
    loss = torch.sum(loss, dim=(-1, -2))[..., None]  # (..., num_samples)
    loss = loss + 1 / 3.0 * torch.sum(weights**2 * (ray_samples.frustums.ends - ray_samples.frustums.starts), dim=-2)

    return loss
