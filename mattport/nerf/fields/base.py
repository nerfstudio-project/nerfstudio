"""
Base class for the graphs.
"""

from abc import abstractmethod

from torch import nn

from mattport.structures.rays import PointSamples, RaySamples


class Field(nn.Module):
    """Base class for fields."""

    def __init__(self) -> None:
        super().__init__()
        # minimum weight for it to be valid. used to skip querying the field
        # TODO(ethan): avoid the hardcoding
        self.weights_threshold = 1e-4

    def density_fn(self, positions):
        """Returns only the density. Used primarily with the occupancy grid."""
        point_samples = PointSamples(positions=positions)
        density, _ = self.get_density(point_samples)
        return density

    @abstractmethod
    def get_density(self, point_samples: PointSamples):
        """Computes and returns the densities."""

    @abstractmethod
    def get_outputs(self, point_samples: PointSamples, density_embedding=None, valid_mask=None):
        """Computes and returns the colors."""

    def forward(self, point_samples: PointSamples):
        """Evaluates the field at points along the ray."""
        density, density_embedding = self.get_density(point_samples)
        field_outputs = self.get_outputs(point_samples, density_embedding=density_embedding)
        field_outputs["density"] = density
        return field_outputs

    def forward_with_weight_pruning(self, ray_samples: RaySamples):
        """Forward but with pruning based on weights, from the densities."""
        density, density_embedding = self.get_density(ray_samples)
        weights = ray_samples.get_weights(density)
        valid_mask_outputs = weights >= self.weights_threshold
        field_outputs = self.get_outputs(
            ray_samples, density_embedding=density_embedding, valid_mask=valid_mask_outputs
        )
        field_outputs["density"] = density
        return field_outputs
