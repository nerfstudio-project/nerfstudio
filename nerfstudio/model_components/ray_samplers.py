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

"""
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import nerfacc
import torch
from nerfacc import OccupancyGrid
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples


class Sampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self) -> RaySamples:
        """Generate Ray Samples"""

    def forward(self, *args, **kwargs) -> RaySamples:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)


class SpacedSampler(Sampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples according to spacing function.

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
        assert num_samples is not None
        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))
        spacing_to_euclidean_fn = lambda x: self.spacing_fn_inv(x * s_far + (1 - x) * s_near)
        euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples


class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LinearDisparitySampler(SpacedSampler):
    """Sample linearly in disparity along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: 1 / x,
            spacing_fn_inv=lambda x: 1 / x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class SqrtSampler(SpacedSampler):
    """Square root sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.sqrt,
            spacing_fn_inv=lambda x: x**2,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class LogSampler(SpacedSampler):
    """Log sampler along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=torch.log,
            spacing_fn_inv=torch.exp,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispPiecewiseSampler(SpacedSampler):
    """Piecewise sampler along a ray that allocates the first half of the samples uniformly and the second half
    using linearly in disparity spacing.


    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.single_jitter = single_jitter

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
            eps: Small value to prevent numerical issues.

        Returns:
            Positions and deltas for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
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
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            ray_samples.spacing_starts is not None and ray_samples.spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert ray_samples.spacing_to_euclidean_fn is not None, "ray_samples.spacing_to_euclidean_fn must be provided"
        existing_bins = torch.cat(
            [
                ray_samples.spacing_starts[..., 0],
                ray_samples.spacing_ends[..., -1:, 0],
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

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )

        return ray_samples


class VolumetricSampler(Sampler):
    """Sampler inspired by the one proposed in the Instant-NGP paper.
    Generates samples along a ray by sampling the occupancy field.
    Optionally removes occluded samples if the density_fn is provided.

    Args:
    occupancy_grid: Occupancy grid to sample from.
    density_fn: Function that evaluates density at a given point.
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    def __init__(
        self,
        occupancy_grid: Optional[OccupancyGrid] = None,
        density_fn: Optional[Callable[[TensorType[..., 3]], TensorType[..., 1]]] = None,
        scene_aabb: Optional[TensorType[2, 3]] = None,
    ) -> None:

        super().__init__()
        self.scene_aabb = scene_aabb
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid
        if self.scene_aabb is not None:
            self.scene_aabb = self.scene_aabb.to("cuda").flatten()
        print(self.scene_aabb)

    def get_sigma_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.

        Args:
            origins: Origins of rays
            directions: Directions of rays
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not self.training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            return density_fn(positions)

        return sigma_fn

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The VolumetricSampler fuses sample generation and density check together. Please call forward() directly."
        )

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        cone_angle: float = 0.0,
    ) -> Tuple[RaySamples, TensorType["total_samples",]]:
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        ray_indices, starts, ends = nerfacc.ray_marching(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            scene_aabb=self.scene_aabb,
            grid=self.occupancy_grid,
            # this is a workaround - using density causes crash and damage quality. should be fixed
            sigma_fn=None,  # self.get_sigma_fn(rays_o, rays_d),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=1e-2,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1, 1), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1, 1), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts,
                ends=ends,
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]
        return ray_samples, ray_indices


class ProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples."""

    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
