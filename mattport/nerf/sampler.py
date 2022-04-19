"""
Collection of sampling strategies
"""
from typing import NamedTuple

import torch
from torch import nn
from torchtyping import TensorType

from mattport.structures.rays import RayBundle, RaySamples


class Sampler(nn.Module):
    """Base Sampler. Intended to be subclassed"""

    def __init__(self, near_plane: float, far_plane: float, num_samples: int) -> None:
        """Inialize sampler

        Args:
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray
        """
        super().__init__()

        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_samples = num_samples

    def generate_samples(
        self, ray_bundle: RayBundle, near_plane: float, far_plane: float, num_samples: int
    ) -> RaySamples:
        """Encodes an input tensor.

        Args:
            camera_rays (cameras.Rays): Rays to generate samples for
            TensorType[..., "output_dim"]: A encoded input tensor
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray

        Returns:
            RaySamples: Positions and deltas for samples along a ray
        """
        raise NotImplementedError

    def forward(self, ray_bundle: RayBundle) -> TensorType[..., "num_samples", 3]:
        """Call forward"""
        return self.generate_samples(
            ray_bundle, near_plane=self.near_plane, far_plane=self.far_plane, num_samples=self.num_samples
        )


class UniformSampler(Sampler):
    """Base Sampler. Intended to be subclassed"""

    def generate_samples(
        self, ray_bundle: RayBundle, near_plane: float, far_plane: float, num_samples: int
    ) -> RaySamples:
        """Encodes an input tensor.

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

        bins = torch.linspace(near_plane, far_plane, num_samples + 1).to(ray_bundle.origins.device)  # (num_samples,)

        num_rays = ray_bundle.origins.shape[0]
        ray_samples = RaySamples(
            t_min=torch.ones_like(ray_bundle.origins[:, 0]) * near_plane,
            t_max=torch.ones_like(ray_bundle.origins[:, 0]) * far_plane,
            ts=bins.unsqueeze(0).repeat(num_rays, 1),  # (num_rays, num_samples)
        )
        return ray_samples
