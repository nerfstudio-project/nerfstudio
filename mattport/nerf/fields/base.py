"""
Base class for the graphs.
"""

from abc import abstractmethod
from typing import Tuple

from torch import nn
from torchtyping import TensorType
from mattport.nerf.field_modules.field_heads import FieldHeadNames

from mattport.structures.rays import PointSamples


class Field(nn.Module):
    """Base class for fields."""

    def density_fn(self, positions):
        """Returns only the density. Used primarily with the occupancy grid."""
        point_samples = PointSamples(positions=positions)
        density, _ = self.get_density(point_samples)
        return density

    @abstractmethod
    def get_density(self, point_samples: PointSamples) -> Tuple[TensorType[..., 1], TensorType[..., "num_features"]]:
        """Computes and returns the densities.

        Args:
            point_samples (PointSamples): Samples locations to compute density.

        Returns:
            Tuple[TensorType[...,1], TensorType[...,"num_features"]]: A tensor of densities and a tensor of features.
        """

    @abstractmethod
    def get_outputs(self, point_samples: PointSamples, density_embedding=None, valid_mask=None):
        """Computes and returns the colors."""

    def forward(self, point_samples: PointSamples):
        """Evaluates the field at points along the ray."""
        density, density_embedding = self.get_density(point_samples)
        field_outputs = self.get_outputs(point_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density
        return field_outputs
