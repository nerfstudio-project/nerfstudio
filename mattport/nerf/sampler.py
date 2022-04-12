"""
Collection of sampling strategies
"""
from typing import NamedTuple

import torch
from torch import nn
from torchtyping import TensorType

from mattport.structures import cameras

RaySamples = NamedTuple(
    "RaySamples", [("locations", TensorType[..., "num_samples", 3]), ("deltas", TensorType[..., "num_samples"])]
)


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
        self, camera_rays: cameras.Rays, near_plane: float, far_plane: float, num_samples: int
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

    def forward(self, camera_rays: cameras.Rays, num_samples: int) -> TensorType[..., "num_samples", 3]:
        """Call forward"""
        return self.Sampler(camera_rays, num_samples)


class UniformSampler(Sampler):
    """Base Sampler. Intended to be subclassed"""

    def generate_samples(
        self, camera_rays: cameras.Rays, near_plane: float, far_plane: float, num_samples: int
    ) -> TensorType[..., "num_samples", 3]:
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

        bins = torch.linspace(near_plane, far_plane, num_samples + 1)

        positions = (bins[..., 1:] + bins[..., :-1]) / 2.0
        deltas = bins[..., 1:] - bins[..., :-1]

        return RaySamples(locations=positions, deltas=deltas)
