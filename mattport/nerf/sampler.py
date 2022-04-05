"""
Collection of sampling strategies
"""
from torchtyping import TensorType
from torch import nn
import torch
import nerf.cameras as cameras


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
    ) -> TensorType[..., "num_samples", 3]:
        """Encodes an input tensor.

        Args:
            camera_rays (cameras.Rays): Rays to generate samples for

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray
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

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
            near_plane (float): Minimum distance along ray to sample
            far_plane (float): Maximum distance along ray to sample
            num_samples (int): Number of samples per ray
        """
        near_plane = near_plane or self.near_plane
        far_plane = far_plane or self.far_plane
        num_samples = num_samples or self.num_samples

        return torch.linspace(near_plane, far_plane, num_samples)
