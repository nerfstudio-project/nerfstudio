# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Shaders for rendering."""
from typing import Optional

from torch import nn
from torchtyping import TensorType


class LambertianShader(nn.Module):
    """Calculate Lambertian shading."""

    @classmethod
    def forward(
        cls,
        rgb: TensorType["bs":..., 3],
        normals: TensorType["bs":..., 3],
        light_direction: TensorType["bs":..., 3],
        shading_weight: float = 1.0,
        detach_normals=True,
    ):
        """Calculate Lambertian shading.

        Args:
            rgb: Accumulated rgb along a ray.
            normals: Accumulated normals along a ray.
            light_direction: Direction of light source.
            shading_weight: Lambertian shading (1.0) vs. ambient lighting (0.0) ratio
            detach_normals: Detach normals from the computation graph when computing shading.

        Returns:
            Textureless Lambertian shading, Lambertian shading
        """
        if detach_normals:
            normals = normals.detach()

        lambertian = (1 - shading_weight) + shading_weight * (normals @ light_direction).clamp(min=0)
        shaded = lambertian.unsqueeze(-1).repeat(1, 3)
        shaded_albedo = rgb * lambertian.unsqueeze(-1)

        return shaded, shaded_albedo


class NormalsShader(nn.Module):
    """Calculate shading for normals."""

    @classmethod
    def forward(
        cls,
        normals: TensorType["bs":..., 3],
        weights: Optional[TensorType["bs":..., 1]] = None,
    ):
        """Applies a rainbow colormap to the normals.

        Args:
            normals: Normalized 3D vectors.
            weights: Optional weights to scale to the normal colors. (Can be used for masking)

        Returns:
            Colored normals
        """
        normals = (normals + 1) / 2
        if weights is not None:
            normals = normals * weights
        return normals
