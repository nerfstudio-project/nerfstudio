# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

from typing import Optional, Union

import torch
from functorch import jacrev, vmap
from torch import nn
from torchtyping import TensorType

from nerfstudio.utils.math import Gaussians


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    def forward(
        self, positions: Union[TensorType["bs":..., 3], Gaussians]
    ) -> Union[TensorType["bs":..., 3], Gaussians]:
        """
        Args:
            positions: Sample to distort

        Returns:
            Union: distorted sample
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 1. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 2.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    def forward(self, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)
            mask = mag >= 1
            x_new = x.clone()
            x_new[mask] = (2 - (1 / mag[mask][..., None])) * (x[mask] / mag[mask][..., None])

            return x_new

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            contract = lambda x: (2 - (1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True))) * (
                x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
            )
            jc_means = vmap(jacrev(contract))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)
