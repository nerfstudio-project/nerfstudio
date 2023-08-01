# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.field_heads import FieldHeadNames


@dataclass
class FieldConfig(InstantiateConfig):
    """Configuration for field instantiation"""

    _target: Type = field(default_factory=lambda: Field)
    """target class to instantiate"""


class Field(nn.Module):
    """Base class for fields."""

    def __init__(self) -> None:
        super().__init__()
        self._sample_locations = None
        self._density_before_activation = None

    def density_fn(
        self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None
    ) -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        del times
        # Need to figure out a better way to describe positions with a ray.
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
    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        """Computes and returns the densities. Returns a tensor of densities and a tensor of features.

        Args:
            ray_samples: Samples locations to compute density.
        """

    def get_normals(self) -> Float[Tensor, "*batch 3"]:
        """Computes and returns a tensor of normals.

        Args:
            density: Tensor of densities.
        """
        assert self._sample_locations is not None, "Sample locations must be set before calling get_normals."
        assert self._density_before_activation is not None, "Density must be set before calling get_normals."
        assert (
            self._sample_locations.shape[:-1] == self._density_before_activation.shape[:-1]
        ), "Sample locations and density must have the same shape besides the last dimension."

        normals = torch.autograd.grad(
            self._density_before_activation,
            self._sample_locations,
            grad_outputs=torch.ones_like(self._density_before_activation),
            retain_graph=True,
        )[0]

        normals = -torch.nn.functional.normalize(normals, dim=-1)

        return normals

    @abstractmethod
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs


def get_normalized_directions(directions: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 3"]:
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0
