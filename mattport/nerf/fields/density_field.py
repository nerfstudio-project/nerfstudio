"""
Density Field.
"""

from torch import nn

from torchtyping import TensorType
from mattport.nerf.field_modules.field_heads import DensityFieldHead
from mattport.structures.rays import RaySamples


class DensityField(nn.Module):
    """Base class for fields."""

    def __init__(self, model: nn.Sequential, density_activation: nn.Module = nn.Softplus()) -> None:
        super().__init__()
        self.model = model
        self.density_head = DensityFieldHead(in_dim=self.model[-1].get_out_dim(), activation=density_activation)

    def process_point(self, in_tensor: TensorType):
        """_summary_"""
        features = self.model(in_tensor)
        density = self.density_head(features)
        return density, features

    def proccess_ray(self, rays: RaySamples):
        """_summary_"""
        densities, features = self.process_point(in_tensor=rays.positions)

        weights = rays.get_weights(densities)
        return densities, weights, features

    def forward(self, in_tensor):
        """summary"""
        return self.process_point(in_tensor)

    def density_fn(self, positions):
        """Returns only the density."""
        density, _ = self.process_point(in_tensor=positions)
        return density
