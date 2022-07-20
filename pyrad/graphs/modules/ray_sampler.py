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

"""
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Callable, Optional

import torch
from torch import nn
from torchtyping import TensorType

from pyrad.cameras.rays import RayBundle, RaySamples
from pyrad.fields.occupancy_fields.occupancy_grid import OccupancyGrid


class Sampler(nn.Module):
    """Generate Samples"""

    def __init__(
        self, num_samples: int, occupancy_field: Optional[OccupancyGrid] = None, weight_threshold: float = 1e-4
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.occupancy_field = occupancy_field
        self.weight_threshold = weight_threshold

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples with optional occupancy filtering"""
        ray_samples = self.generate_ray_samples(*args, **kwargs)
        if self.occupancy_field is not None:
            densities = self.occupancy_field.get_densities(ray_samples.frustums.get_positions())
            weights = ray_samples.get_weights(densities)

            valid_mask = weights >= self.weight_threshold
            ray_samples.set_valid_mask(valid_mask)
        return ray_samples


class SpacedSampler(Sampler):
    """Sample points according to a function."""

    def __init__(
        self,
        num_samples: int,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        train_stratified=True,
        occupancy_field: Optional[OccupancyGrid] = None,
        weight_threshold: float = 1e-4,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            spacing_fn (Callable): Function that dictates sample spacing (ie `lambda x : x` is uniform).
            spacing_fn_inv (Callable): The inverse of spacing_fn.
            train_stratified (bool): Use stratified sampling during training. Defults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
        """
        super().__init__(num_samples=num_samples, occupancy_field=occupancy_field, weight_threshold=weight_threshold)
        self.train_stratified = train_stratified
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples accoring to spacing function.

        Args:
            ray_bundle (RayBundle): Rays to generate samples for
            num_samples (Optional[int]): Number of samples per ray

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        s_near, s_far = [self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars)]
        bins = self.spacing_fn_inv(bins * s_far + (1 - bins) * s_near)  # [num_rays, num_samples+1]

        if self.train_stratified and self.training:
            t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        ray_samples = ray_bundle.get_ray_samples(bin_starts=bins[..., :-1, None], bin_ends=bins[..., 1:, None])

        return ray_samples


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray"""

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        occupancy_field: Optional[OccupancyGrid] = None,
        weight_threshold: float = 1e-4,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified (bool): Use stratified sampling during training. Defults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
        """
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            occupancy_field=occupancy_field,
            weight_threshold=weight_threshold,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray"""

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        occupancy_field: Optional[OccupancyGrid] = None,
        weight_threshold: float = 1e-4,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified (bool): Use stratified sampling during training. Defults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
        """
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            occupancy_field=occupancy_field,
            weight_threshold=weight_threshold,
        )


class SqrtSampler(SpacedSampler):
    """Square root sampler along a ray"""

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        occupancy_field: Optional[OccupancyGrid] = None,
        weight_threshold: float = 1e-4,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified (bool): Use stratified sampling during training. Defults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
        """
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            occupancy_field=occupancy_field,
            weight_threshold=weight_threshold,
        )


class LogSampler(SpacedSampler):
    """Log sampler along a ray"""

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        occupancy_field: Optional[OccupancyGrid] = None,
        weight_threshold: float = 1e-4,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified (bool): Use stratified sampling during training. Defults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
        """
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            occupancy_field=occupancy_field,
            weight_threshold=weight_threshold,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution"""

    def __init__(
        self,
        num_samples: int,
        train_stratified: bool = True,
        include_original: bool = True,
        occupancy_field: OccupancyGrid = None,
        weight_threshold: float = 1e-4,
        histogram_padding: float = 0.01,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified: boolean: Randomize location within each bin during training. Defaults to True
            include_original: Add original samples to ray. Defaults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
            histogram_padding (float): Amount to weights prior to computing PDF. Defaults to 0.01.
        """
        super().__init__(num_samples=num_samples, occupancy_field=occupancy_field, weight_threshold=weight_threshold)
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle = None,
        ray_samples: RaySamples = None,
        weights: TensorType[..., "num_samples", 1] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle (RayBundle): Rays to generate samples for
            ray_samples (RaySamples): Existing ray samples
            weights: (TensorType[..., "num_samples", 1]): Weights for each bin
            num_samples (Optional[int]): Number of samples per ray
            eps: float: Small value to prevent numerical issues. Defaults to 1e-5

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        num_samples = num_samples or self.num_samples
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            u = u + torch.rand(size=(*cdf.shape[:-1], num_bins), device=cdf.device) / num_bins
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0, steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        # Force bins to not have a gap between them. Kinda hacky, should reconsider.
        existing_bins = torch.cat(
            [
                ray_samples.frustums.starts[..., :1, 0],
                (ray_samples.frustums.starts[..., 1:, 0] + ray_samples.frustums.ends[..., :-1, 0]) / 2.0,
                ray_samples.frustums.ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        ray_samples = ray_bundle.get_ray_samples(bin_starts=bins[..., :-1, None], bin_ends=bins[..., 1:, None])

        return ray_samples
