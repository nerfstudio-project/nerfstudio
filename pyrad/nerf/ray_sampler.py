"""
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType

from pyrad.nerf.occupancy_grid import OccupancyGrid
from pyrad.structures.rays import RayBundle, RaySamples


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
            densities = self.occupancy_field.get_densities(ray_samples.positions)
            weights = ray_samples.get_weights(densities)

            valid_mask = weights >= self.weight_threshold
            ray_samples.set_valid_mask(valid_mask)
        return ray_samples


class UniformSampler(Sampler):
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
        super().__init__(num_samples=num_samples, occupancy_field=occupancy_field, weight_threshold=weight_threshold)
        self.train_stratified = train_stratified

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples uniformly.

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

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)  # shape (num_samples+1,)
        bins = ray_bundle.nears[:, None] + bins[None, :] * (
            ray_bundle.fars[:, None] - ray_bundle.nears[:, None]
        )  # shape (num_rays, num_samples+1)

        if self.train_stratified and self.training:
            t_rand = torch.rand((num_rays, num_samples), dtype=bins.dtype, device=bins.device)
            ts = bins[:, :-1] + t_rand * (bins[:, 1:] - bins[:, :-1])  # shape (num_rays, num_samples)
        else:
            ts = (bins[:, 1:] + bins[:, :-1]) / 2

        ray_samples = ray_bundle.get_ray_samples(ts)

        return ray_samples


class PDFSampler(Sampler):
    """Sample based on probability distribution"""

    def __init__(
        self,
        num_samples: int,
        train_stratified: bool = True,
        include_original: bool = True,
        occupancy_field: OccupancyGrid = None,
        weight_threshold: float = 1e-4,
    ) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified: boolean: Randomize location within each bin during training. Defaults to True
            include_original: Add original samples to ray. Defaults to True
            occupancy_field (OccupancyGrid, optional): Occupancy grid. If provides,
                samples below weight_threshold as set as invalid.
            weight_thershold (float): Removes samples below threshold weight. Only used if occupancy field is provided.
        """
        super().__init__(num_samples=num_samples, occupancy_field=occupancy_field, weight_threshold=weight_threshold)
        self.train_stratified = train_stratified
        self.include_original = include_original

    @torch.no_grad()
    def generate_ray_samples(
        self,
        ray_bundle: RayBundle = None,
        ray_samples: RaySamples = None,
        weights: TensorType[..., "num_samples"] = None,
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle (RayBundle): Rays to generate samples for
            ray_samples (RaySamples): Existing ray samples
            weights: (TensorType[..., "num_samples"]): Weights for each bin
            num_samples (Optional[int]): Number of samples per ray
            eps: float: Small value to prevent numerical issues. Defaults to 1e-5

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        num_samples = num_samples or self.num_samples
        weights = weights[..., :-1]

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and self.training:
            u = torch.rand(size=(*cdf.shape[:-1], num_samples), device=cdf.device)
        else:
            u = torch.linspace(0.0, 1.0, steps=num_samples, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_samples))

        u = u.contiguous()
        indicies = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(indicies), indicies - 1)
        above = torch.min(cdf.shape[-1] - torch.ones_like(indicies), indicies)

        indicies_g = torch.stack([below, above], -1)
        matched_shape = (indicies_g.shape[0], indicies_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), dim=2, index=indicies_g)
        bins_g = torch.gather(ray_samples.ts.unsqueeze(1).expand(matched_shape), dim=2, index=indicies_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < eps, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        ts = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        if self.include_original:
            ts, _ = torch.sort(torch.cat([ray_samples.ts, ts], -1), -1)
        else:
            ts, _ = torch.sort(ts, -1)

        # Stop gradients
        ts = ts.detach()

        ray_samples = ray_bundle.get_ray_samples(ts)

        return ray_samples
