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
"""Cuda toolbox corresponding to `rays.py`."""
from dataclasses import dataclass

import torch
from torchtyping import TensorType

from pyrad.cameras.rays import Frustums, RaySamples


@dataclass
class RaySamplesPacked(RaySamples):
    """Packed samples along ray

    Args:
        frustums (Frustums): Frustums along ray.
        camera_indices (TensorType[..., 1]): Camera index.
        valid_mask (TensorType[..., 1]): Samples that are valid.
        deltas (TensorType[..., 1]): "width" of each sample.
        packed_indices (TensorType["num_rays", 2]): Indices for recovering rays from
            packed samples. Each ray has two integer values recorded here. The first
            value records the start index of the samples, and the second value records
            number of samples for this ray (a.k.a shift in the packed samples between
            this ray and the next ray).
    """

    frustums: Frustums
    camera_indices: TensorType["packed_samples", 1] = None
    valid_mask: TensorType["packed_samples", 1] = None
    deltas: TensorType["packed_samples", 1] = None
    packed_indices: TensorType["num_rays", 3] = None

    def get_weights(self, densities: TensorType["packed_samples", 1]) -> TensorType["packed_samples", 1]:
        """Return weights based on predicted densities

        Args:
            densities (TensorType["packed_samples", 1]): Predicted densities for samples along ray

        Returns:
            TensorType["packed_samples", 1]: Weights for each sample
        """
        raise NotImplementedError

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        # mip-nerf version of transmittance calculation:
        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1)).to(densities.device), transmittance], axis=-2
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        # most nerf codebases do the following:
        # transmittance = torch.cat(
        #     [torch.ones((*alphas.shape[:1], 1)).to(densities.device), 1.0 - alphas + 1e-10], dim=-1
        # )
        # transmittance = torch.cumprod(transmittance, dim=-1)[..., :-1]  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        return weights

    def set_valid_mask(self, **_) -> None:  # pylint: disable=arguments-differ
        raise ValueError("this function should not be called!")

    def apply_masks(self) -> "RaySamples":
        raise ValueError("this function should not be called!")
