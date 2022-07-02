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

"""Space distortions."""

from typing import Union

import torch
from functorch import jacrev, vmap
from torch import nn
from torchtyping import TensorType

from pyrad.utils.math import Gaussians


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    def forward(self, positions: Union[TensorType[..., 3], Gaussians]) -> Union[TensorType[..., 3], Gaussians]:
        """
        Args:
            positions (Union[TensorType[..., 3], Gaussians]): Sample to distort

        Returns:
            Union[TensorType[..., 3], Gaussians]: distorted sample
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space into a sphere of radius 2. This contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}
    """

    def forward(self, positions):
        def contract(x):
            mag = x.norm(dim=-1)
            mask = mag >= 1
            x[mask] = (2 - (1 / mag[mask][..., None])) * (x[mask] / mag[mask][..., None])

            return x

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            contract = lambda x: (2 - (1 / x.norm(dim=-1, keepdim=True))) * (x / x.norm(dim=-1, keepdim=True))
            jc_means = vmap(jacrev(contract))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)
