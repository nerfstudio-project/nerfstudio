"""
Collection of sampling strategies
"""

from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from pyrad.fields.occupancy_fields.occupancy_grid import OccupancyGrid
from pyrad.cameras.rays import RayBundle, RaySamples


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

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)  # shape (num_samples+1)
        bins = ray_bundle.nears[:, None] + bins[None, :] * (
            ray_bundle.fars[:, None] - ray_bundle.nears[:, None]
        )  # shape (num_rays, num_samples+1)

        if self.train_stratified and self.training:
            t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand

        ray_samples = ray_bundle.get_ray_samples(bins)

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
        num_bins = num_samples + 1

        weights = weights + self.histogram_padding

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

        mask = u[..., None, :] >= cdf[..., :, None]  # [num_samples, num_orig_bins, num_bins]

        # Uses same interval trick as mip-NeRF
        def find_interval(
            mask: TensorType[..., "num_orig_bins", "num_bins"], x: TensorType[..., "num_orig_bins"]
        ) -> Tuple[TensorType[..., "num_bins"], TensorType[..., "num_bins"]]:
            """Find intervals based on cdf mask.

                 Mask                x              x_start         x_end
               T T T T F      [x0 x1 x2 x3 x4]   [x0 x2 x3 x3]   [x1 x3 x4 x4]
               T T T T F
               T T T F F
               T F F F F

               Where the number of rows correspond to the target number of bins. The number of columns
               correspond to the input number of bins

            Args:
                mask (TensorType[..., "num_orig_bins", "num_bins"]): PDF represented as a boolean mask.
                x (TensorType[..., "num_original_bins"]): Probe to calculate intervals for.

            Returns:
                Tuple[TensorType[..., "num_bins"], TensorType[..., "num_bins"]]: (x_start, x_end)
            """

            x_start = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)[0]
            x_end = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)[0]
            return x_start, x_end

        bins_g0, bins_g1 = find_interval(mask, ray_samples.bins)
        cdf_g0, cdf_g1 = find_interval(mask, cdf)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([ray_samples.bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        ray_samples = ray_bundle.get_ray_samples(bins)

        return ray_samples
