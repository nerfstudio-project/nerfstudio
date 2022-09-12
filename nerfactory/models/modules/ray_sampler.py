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
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

import nerfactory.cuda as nerfactory_cuda
from nerfactory.cameras.rays import Frustums, RayBundle, RaySamples
from nerfactory.fields.density_fields.density_grid import DensityGrid


class Sampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
        density_field: density grid specifying weighting of
    """

    def __init__(
        self, num_samples: int, density_field: Optional[DensityGrid] = None, weight_threshold: float = 0.01
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.density_field = density_field
        self.weight_threshold = weight_threshold

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples with optional density filtering"""
        ray_samples = self.generate_ray_samples(*args, **kwargs)
        if self.density_field is not None:
            densities = self.density_field.get_densities(ray_samples)
            deltas = torch.clamp(ray_samples.frustums.ends - ray_samples.frustums.starts, min=1e-10)
            density_threshold = torch.clamp(self.weight_threshold / deltas, max=self.density_field.mean_density)
            valid_mask = densities > density_threshold
            ray_samples.set_valid_mask(valid_mask & ray_samples.valid_mask)
        return ray_samples


class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defults to True
        density_field: Density grid. If provides, samples below weight_threshold as set as invalid.
        weight_threshold: Removes samples below threshold weight. Only used if density field is provided.
    """

    def __init__(
        self,
        num_samples: int,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        train_stratified=True,
        density_field: Optional[DensityGrid] = None,
        weight_threshold: float = 1e-2,
    ) -> None:
        super().__init__(num_samples=num_samples, density_field=density_field, weight_threshold=weight_threshold)
        self.train_stratified = train_stratified
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples accoring to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.nears is not None
        assert ray_bundle.fars is not None

        num_samples = num_samples or self.num_samples
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))
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
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        density_field: Density grid. If provides, samples below weight_threshold as set as invalid.
        weight_threshold: Removes samples below threshold weight. Only used if density field is provided.
    """

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        density_field: Optional[DensityGrid] = None,
        weight_threshold: float = 1e-2,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            density_field=density_field,
            weight_threshold=weight_threshold,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        density_field: Density grid. If provides, samples below weight_threshold as set as invalid.
        weight_threshold: Removes samples below threshold weight. Only used if density field is provided.
    """

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        density_field: Optional[DensityGrid] = None,
        weight_threshold: float = 1e-2,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            density_field=density_field,
            weight_threshold=weight_threshold,
        )


class SqrtSampler(SpacedSampler):
    """Square root sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        density_field: Density grid. If provides, samples below weight_threshold as set as invalid.
        weight_threshold: Removes samples below threshold weight. Only used if density field is provided.
    """

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        density_field: Optional[DensityGrid] = None,
        weight_threshold: float = 1e-2,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            density_field=density_field,
            weight_threshold=weight_threshold,
        )


class LogSampler(SpacedSampler):
    """Log sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defults to True
        density_field: Density grid. If provides, samples below weight_threshold as set as invalid.
        weight_threshold: Removes samples below threshold weight. Only used if density field is provided.
    """

    def __init__(
        self,
        num_samples: int,
        train_stratified=True,
        density_field: Optional[DensityGrid] = None,
        weight_threshold: float = 1e-2,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            density_field=density_field,
            weight_threshold=weight_threshold,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training. Defaults to True
        include_original: Add original samples to ray. Defaults to True
        density_field: Density grid. If provides, samples below weight_threshold as set as invalid.
        weight_threshold: Removes samples below threshold weight. Only used if density field is provided.
        histogram_padding: Amount to weights prior to computing PDF. Defaults to 0.01.
    """

    def __init__(
        self,
        num_samples: int,
        train_stratified: bool = True,
        include_original: bool = True,
        density_field: Optional[DensityGrid] = None,
        weight_threshold: float = 1e-2,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__(num_samples=num_samples, density_field=density_field, weight_threshold=weight_threshold)
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        ray_samples: Optional[RaySamples] = None,
        weights: TensorType[..., "num_samples", 1] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray
            eps: Small value to prevent numerical issues. Defaults to 1e-5

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")

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


class NGPSpacedSampler(Sampler):
    """Sampler that matches Instant-NGP paper.

    This sampler does ray-box AABB test, ray samples generation and density check, all together.

    TODO(ruilongli): check whether fuse AABB test together with `raymarching` can speed things up.
    Otherwise we can seperate it out and use collision detecoter that already exists in the repo.
    """

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError("For NGP we fused ray samples and density check together. Please call forward() directly.")

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        aabb: TensorType[2, 3],
        num_samples: Optional[int] = None,
        marching_steps: Optional[int] = 128,
        t_min: Optional[TensorType["num_rays"]] = None,
        t_max: Optional[TensorType["num_rays"]] = None,
    ) -> Tuple[RaySamples, TensorType["total_samples", 3], TensorType["total_samples"], TensorType["total_samples"]]:
        """Generate ray samples in a bounding box.

        TODO(ruilongli): write a Packed[Ray_samples] class to ray_samples with packed_info.
        TODO(ruilongli): maybe move aabb test to the collision detector?

        Args:
            ray_bundle: Rays to generate samples for
            aabb: Bounding box of the scene.
            num_samples: Number of samples per ray

        Returns:
            First return is ray samples in a packed way where only the valid samples are kept.
            Second return contains all the information to recover packed samples into unpacked mode for rendering.
            The last two returns are t_min and t_max from ray-aabb test.
        """

        if self.density_field is None:
            raise ValueError("density_field must be set to use NGPSpacedSampler")

        num_samples = num_samples or self.num_samples

        aabb = aabb.flatten()
        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        scene_scale = (aabb[3:6] - aabb[0:3]).max()

        if t_min is None or t_max is None:
            # TODO(ruilongli): this clipping here is stupid. Try to avoid that.
            t_min, t_max = nerfactory_cuda.ray_aabb_intersect(rays_o, rays_d, aabb)
            t_min = torch.clamp(t_min, max=1e10)  # type: ignore
            t_max = torch.clamp(t_max, max=1e10)  # type: ignore

        marching_steps = -1  # disable marching mode for now
        # TODO(ruilongli): * 16 is for original impl for training. Need to run
        # some profiling test with this choice.
        max_samples_per_batch = len(rays_o) * num_samples

        # marching
        packed_info, origins, dirs, starts, ends = nerfactory_cuda.raymarching(
            # rays
            rays_o,
            rays_d,
            t_min,
            t_max,
            # density grid
            self.density_field.center,
            self.density_field.base_scale,
            self.density_field.num_cascades,
            self.density_field.resolution,
            self.density_field.density_bitfield,
            # sampling args
            marching_steps,
            max_samples_per_batch,
            num_samples,
            0.0,
            scene_scale,
        )

        # squeeze valid samples
        total_samples = max(packed_info[:, -1].sum(), 1)
        origins = origins[:total_samples]
        dirs = dirs[:total_samples]
        starts = starts[:total_samples]
        ends = ends[:total_samples]

        # return samples
        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(origins=origins, directions=dirs, starts=starts, ends=ends, pixel_area=zeros),
        )
        return ray_samples, packed_info, t_min, t_max
