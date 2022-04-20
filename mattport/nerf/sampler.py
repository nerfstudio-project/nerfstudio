"""
Collection of sampling strategies
"""

import torch
from torch import nn
from torchtyping import TensorType

from mattport.structures.rays import RayBundle, RaySamples


class UniformSampler(nn.Module):
    """Sample uniformly along a ray"""

    def forward(
        self, camera_rays: cameras.Rays, near_plane: float, far_plane: float, num_samples: int
    ) -> TensorType[..., "num_samples", 3]:
        """Generates position samples uniformly.

        Args:
            camera_rays (cameras.Rays): Rays to generate samples for
            TensorType[..., "output_dim"]: A encoded input tensor
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        near_plane = near_plane or self.near_plane
        far_plane = far_plane or self.far_plane
        num_samples = num_samples or self.num_samples

        bins = torch.linspace(near_plane, far_plane, num_samples + 1)

        t = (bins[..., 1:] + bins[..., :-1]) / 2.0
        deltas = bins[..., 1:] - bins[..., :-1]

        positions = camera_rays.origin + camera_rays.direction * t

        return RaySamples(locations=positions, deltas=deltas)


class PDFSampler(nn.Module):
    """Sample based on probability distribution"""

    def forward(
        self,
        camera_rays: cameras.Rays,
        num_samples: int,
        bins: TensorType[..., "num_samples"],
        weights: TensorType[..., "num_samples"],
        randomized: bool = True,
        eps: float = 1e-5,
    ) -> TensorType[..., "num_samples", 3]:
        """Generates position samples given a distribution.

        Args:
            camera_rays (cameras.Rays): Rays to generate samples for
            TensorType[..., "output_dim"]: A encoded input tensor
            num_samples (int): Number of samples per ray
            bins (TensorType[..., "num_samples"]): Ray bins
            weights: (TensorType[..., "num_samples"]): Weights for each bin
            randomized: boolean: Randomize location within each bin. Defaults to True
            eps: float: Small value to prevent numerical issues. Defaults to 1e-5

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights += padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(1, torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if randomized:
            u = torch.rand(size=(*cdf.shape[:-1], num_samples))
        else:
            u = torch.linspace(0.0, 1.0, steps=num_samples)
            u = torch.expand(size=(*cdf.shape[:-1], num_samples))

        # u = u.contigous()
        indicies = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(indicies), indicies - 1)
        above = torch.min(cdf.shape[-1] - 1, indicies)

        indicies_g = torch.stack([below, above], -1)
        cdf_g = torch.gather(cdf, indicies_g, axis=-1, batch_dims=len(indicies_g.shape) - 2)
        bins_g = torch.gather(bins, indicies_g, axis=-1, batch_dims=len(indicies_g.shape) - 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < eps, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        deltas = samples[..., 1:] - samples[..., :-1]

        positions = camera_rays.origin + camera_rays.direction * t

        num_rays = ray_bundle.origins.shape[0]
        ray_samples = RaySamples(
            t_min=torch.ones_like(ray_bundle.origins[:, 0]) * near_plane,
            t_max=torch.ones_like(ray_bundle.origins[:, 0]) * far_plane,
            ts=bins.unsqueeze(0).repeat(num_rays, 1),  # (num_rays, num_samples)
            ray_bundle=ray_bundle,
        )
        return ray_samples
