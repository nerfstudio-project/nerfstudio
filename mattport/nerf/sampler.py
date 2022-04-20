"""
Collection of sampling strategies
"""

from typing import Optional
import torch
from torch import nn
from torchtyping import TensorType

from mattport.structures.rays import RayBundle, RaySamples


class UniformSampler(nn.Module):
    """Sample uniformly along a ray"""

    def __init__(self, near_plane: float, far_plane: float, num_samples: int) -> None:
        """
        Args:
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray
        """
        super().__init__()

        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_samples = num_samples

    def forward(
        self,
        ray_bundle: RayBundle,
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> RaySamples:
        """Generates position samples uniformly.

        Args:
            ray_bundle (RayBundle): Rays to generate samples for
            TensorType[..., "output_dim"]: A encoded input tensor
            near_plane (Optional[float]): Minimum distance along ray to sample
            far_plane (Optional[float]): Maximum distance along ray to sample
            num_samples (Optional[int]): Number of samples per ray

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        near_plane = near_plane or self.near_plane
        far_plane = far_plane or self.far_plane
        num_samples = num_samples or self.num_samples

        num_rays = ray_bundle.origins.shape[0]

        bins = torch.linspace(near_plane, far_plane, num_samples + 1).to(ray_bundle.origins.device)  # (num_samples+1,)
        bins = bins.unsqueeze(0).repeat(num_rays, 1)  # (num_rays, num_samples+1)

        ray_samples = RaySamples(
            bins=bins,
            ray_bundle=ray_bundle,
        )

        return ray_samples


class PDFSampler(nn.Module):
    """Sample based on probability distribution"""

    def __init__(self, num_samples: int) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
        """
        super().__init__()

        self.num_samples = num_samples

    def forward(
        self,
        ray_bundle: RayBundle,
        coarse_ray_samples: RaySamples,
        density: TensorType[..., "num_samples", 1],
        num_samples: Optional[int] = None,
        randomized: bool = True,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle (RayBundle): Rays to generate samples for
            TensorType[..., "output_dim"]: A encoded input tensor
            bins (TensorType[..., "num_samples"]): Ray bins
            weights: (TensorType[..., "num_samples"]): Weights for each bin
            num_samples (Optional[int]): Number of samples per ray
            randomized: boolean: Randomize location within each bin. Defaults to True
            eps: float: Small value to prevent numerical issues. Defaults to 1e-5

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """

        num_samples = num_samples or self.num_samples

        # Calculate weight contributions along ray
        # Todo(matt): This computation is duplicated
        delta_density = coarse_ray_samples.deltas * density[..., 0]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1], dim=-1)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1)).to(density.device), transmittance], axis=-1
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
        weights = alphas * transmittance  # [..., "num_samples"]

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights += padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if randomized:
            u = torch.rand(size=(*cdf.shape[:-1], num_samples), device=cdf.device)
        else:
            u = torch.linspace(0.0, 1.0, steps=num_samples, device=cdf.device)
            u = torch.expand(size=(*cdf.shape[:-1], num_samples))

        # u = u.contiguous()
        indicies = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(indicies), indicies - 1)
        above = torch.min(cdf.shape[-1] - torch.ones_like(indicies), indicies)

        indicies_g = torch.stack([below, above], -1)
        matched_shape = (indicies_g.shape[0], indicies_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), dim=2, index=indicies_g)
        bins_g = torch.gather(coarse_ray_samples.bins.unsqueeze(1).expand(matched_shape), dim=2, index=indicies_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < eps, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        ray_samples = RaySamples(
            bins=samples,  # TODO(matt) These are not bins!
            ray_bundle=ray_bundle,
        )

        return ray_samples
