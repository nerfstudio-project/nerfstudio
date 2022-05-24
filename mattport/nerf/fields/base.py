"""
Base class for the graphs.
"""

from abc import abstractmethod
from typing import Optional, Tuple

from torch import nn
import torch
from torchtyping import TensorType
from mattport.nerf.field_modules.field_heads import FieldHeadNames

from mattport.structures.rays import PointSamples
from mattport.utils.misc import is_not_none


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
    def get_outputs(
        self, point_samples: PointSamples, density_embedding: Optional[TensorType] = None
    ) -> dict[FieldHeadNames, TensorType]:
        """Computes and returns the colors.

        Args:
            point_samples (PointSamples): Samples locations to compute outputs.
            density_embedding (TensorType, optional): Density embeddings to condition on.

        Returns:
            Dict[FieldHeadNames, TensorType]: Output field values.
        """

    def forward(self, point_samples: PointSamples):
        """Evaluates the field at points along the ray."""
        valid_mask = point_samples.valid_mask

        if is_not_none(valid_mask):
            # Hacky handling of empty masks. Tests on a single ray but doesn't use results
            if not valid_mask.any():
                point_samples = PointSamples(
                    positions=point_samples.positions[0, :], directions=point_samples.directions[0, :]
                )
            else:
                point_samples = point_samples.apply_masks()
            density_masked, density_embedding_masked = self.get_density(point_samples)
            field_outputs_masked = self.get_outputs(point_samples, density_embedding=density_embedding_masked)

            field_outputs = {}
            for k, value in field_outputs_masked.items():
                zeros = torch.zeros(*valid_mask.shape, value.shape[-1], dtype=torch.float32, device=valid_mask.device)
                if valid_mask.any():
                    zeros[valid_mask] = value
                else:
                    zeros[0, :] = value
                field_outputs[k] = zeros
            density = torch.zeros(*valid_mask.shape, 1, dtype=torch.float32, device=valid_mask.device)
            if valid_mask.any():
                density[valid_mask] = density_masked
        else:
            density, density_embedding = self.get_density(point_samples)
            field_outputs = self.get_outputs(point_samples, density_embedding=density_embedding)

        field_outputs[FieldHeadNames.DENSITY] = density
        return field_outputs
