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

    def __init__(self, near_plane: float, far_plane: float, num_samples: int, train_stratified=True) -> None:
        """
        Args:
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray
            train_stratified (bool): Use stratified sampling during training. Defults to True
        """
        super().__init__()

        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_samples = num_samples
        self.train_stratified = train_stratified

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

        bins = torch.linspace(near_plane, far_plane, num_samples + 1).to(ray_bundle.origins.device)  # (num_samples+1)

        if self.train_stratified and self.training:
            t_rand = torch.rand((num_rays, num_samples), dtype=bins.dtype, device=bins.device)
            ts = bins[None, :-1] + t_rand * (bins[None, 1:] - bins[None, :-1])  # (num_rays, num_samples)
        else:
            ts = (bins[1:] + bins[:-1]) / 2
            ts = ts.unsqueeze(0).repeat(num_rays, 1)  # (num_rays, num_samples)

        ray_samples = RaySamples(
            ts=ts,
            ray_bundle=ray_bundle,
        )

        return ray_samples


class PDFSampler(nn.Module):
    """Sample based on probability distribution"""

    def __init__(self, num_samples: int, train_stratified: bool = True, include_original: bool = True) -> None:
        """
        Args:
            num_samples (int): Number of samples per ray
            train_stratified: boolean: Randomize location within each bin during training. Defaults to True
            include_original: Add original samples to ray. Defaults to True
        """
        super().__init__()

        self.num_samples = num_samples
        self.include_original = include_original
        self.train_stratified = train_stratified

    def forward(
        self,
        coarse_ray_samples: RaySamples,
        weights: TensorType[..., "num_samples"],
        num_samples: Optional[int] = None,
        eps: float = 1e-5,
    ) -> RaySamples:
        """Generates position samples given a distribution.

        Args:
            ray_bundle (RayBundle): Rays to generate samples for
            TensorType[..., "output_dim"]: A encoded input tensor
            bins (TensorType[..., "num_samples"]): Ray bins
            weights: (TensorType[..., "num_samples"]): Weights for each bin
            num_samples (Optional[int]): Number of samples per ray
            eps: float: Small value to prevent numerical issues. Defaults to 1e-5

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        # TODO (matt) Look into torch.no_grad(), or torch.inference_mode

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
            u = torch.expand(size=(*cdf.shape[:-1], num_samples))

        u = u.contiguous()
        indicies = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(indicies), indicies - 1)
        above = torch.min(cdf.shape[-1] - torch.ones_like(indicies), indicies)

        indicies_g = torch.stack([below, above], -1)
        matched_shape = (indicies_g.shape[0], indicies_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), dim=2, index=indicies_g)
        bins_g = torch.gather(coarse_ray_samples.ts.unsqueeze(1).expand(matched_shape), dim=2, index=indicies_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < eps, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        ts = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        if self.include_original:
            ts, _ = torch.sort(torch.cat([coarse_ray_samples.ts, ts], -1), -1)
        else:
            ts, _ = torch.sort(ts, -1)

        # Stop gradients
        ts = ts.detach()

        ray_samples = RaySamples(
            ts=ts,
            ray_bundle=coarse_ray_samples.ray_bundle,
        )

        return ray_samples
