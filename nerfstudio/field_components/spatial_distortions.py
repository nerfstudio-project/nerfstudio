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

"""Space distortions."""

import abc
from typing import Optional, Union, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.utils.math import Gaussians


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    @abc.abstractmethod
    def forward(self, positions: Union[Float[Tensor, "*bs 3"], Gaussians]) -> Union[Float[Tensor, "*bs 3"], Gaussians]:
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
        radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    def forward(self, positions):
        def contract(x: torch.Tensor) -> torch.Tensor:
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            def contract_gauss(x: torch.Tensor) -> torch.Tensor:
                mag = torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
                return (2 - 1 / mag) * (x / mag)

            jc_means = torch.func.vmap(torch.func.jacrev(contract_gauss))(
                positions.mean.view(-1, positions.mean.shape[-1])
            )
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = (jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)).to(cov)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)


class LinearizedSceneContraction(SpatialDistortion):
    """Lightweight contract unbounded space using the contraction was proposed in ZipNeRF.

    Proposed only for Gausians. Part of code was taken from github.com/SuLvXiangXin/zipnerf-pytorch
    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    def forward(self, positions: Gaussians) -> Gaussians:
        def contract(
            mean: Float[Tensor, "*bs 3"],
            cov: Float[Tensor, "*bs"],
            eps: float = 1e-7,
        ) -> Tuple[Float[Tensor, "*bs 3"], Float[Tensor, "*bs"]]:
            """ZipNeRF contraction function"""
            mag = torch.linalg.norm(mean, ord=self.order, dim=-1, keepdim=False).clamp_min(eps)
            mask = mag < 1
            mean_contracted = torch.where(mask[..., None], mean, (2 - (1 / mag[..., None])) * (mean / mag[..., None]))
            # prevent negative root computations
            clamped_mag = mag.clamp_min(1.0)
            det_13 = (torch.pow(2 * clamped_mag - 1, 1 / 3) / clamped_mag) ** 2
            std_contracted = torch.where(mask, cov, cov * det_13)

            return mean_contracted, std_contracted

        assert isinstance(positions, Gaussians)

        pre_shape = positions.mean.shape[:-1]

        means = positions.mean.view(-1, 3)
        cov = positions.cov.view(-1)

        means, cov = contract(means, cov)

        means = means.view(*pre_shape, 3)
        cov = cov.view(*pre_shape)

        return Gaussians(mean=means, cov=cov)
