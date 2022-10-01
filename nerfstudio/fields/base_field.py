# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base class for the graphs.
"""

from abc import abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames


class Field(nn.Module):
    """Base class for fields."""

    def density_fn(self, positions: TensorType["bs":..., 3]) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            )
        )
        density, _ = self.get_density(ray_samples)
        return density

    @abstractmethod
    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType[..., 1], TensorType[..., "num_features"]]:
        """Computes and returns the densities. Returns a tensor of densities and a tensor of features.

        Args:
            ray_samples: Samples locations to compute density.
        """

    @abstractmethod
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        valid_mask = ray_samples.valid_mask

        if valid_mask is not None:
            # Hacky handling of empty masks. Tests on a single ray but doesn't use results
            if not valid_mask.any():
                # TODO: Can we just throw an exception here? If not we need to update this to include some fake metadata
                ray_samples = RaySamples(frustums=Frustums.get_mock_frustum().to(valid_mask.device))
            else:
                ray_samples = ray_samples.apply_masks()
            density_masked, density_embedding_masked = self.get_density(ray_samples)
            field_outputs_masked = self.get_outputs(ray_samples, density_embedding=density_embedding_masked)

            field_outputs = {}
            for k, value in field_outputs_masked.items():
                zeros = torch.zeros(
                    *valid_mask.shape[:-1], value.shape[-1], dtype=value.dtype, device=valid_mask.device
                )
                if valid_mask.any():
                    zeros[valid_mask[..., 0]] = value
                else:
                    zeros[0, :] = value
                field_outputs[k] = zeros
            density = torch.zeros(valid_mask.shape, dtype=density_masked.dtype, device=valid_mask.device)
            if valid_mask.any():
                density[valid_mask[..., 0]] = density_masked
        else:
            density, density_embedding = self.get_density(ray_samples)
            field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)

        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        return field_outputs
