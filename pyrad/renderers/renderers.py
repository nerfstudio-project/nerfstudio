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
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import math
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType
from pyrad.cameras.rays import RaySamples

from pyrad.utils.math import components_from_spherical_harmonics


class RGBRenderer(nn.Module):
    """Standard volumetic rendering.

    Args:
        background_color: Background color as RGB. Defaults to random.
    """

    def __init__(self, background_color: Optional[TensorType[3]] = None) -> None:
        super().__init__()
        self.background_color = background_color

    @classmethod
    def combine_rgb(
        cls,
        rgb: TensorType[..., "num_samples", 3],
        weights: TensorType[..., "num_samples", 1],
        background_color: Optional[TensorType[3]] = None,
    ) -> TensorType[..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB. Defaults to random.

        Returns:
            Outputs rgb values.
        """
        rgb = torch.sum(weights * rgb, dim=-2)

        if background_color is None:
            background_color = torch.rand_like(rgb).to(rgb.device)

        rgb = rgb + background_color.to(weights.device) * (1.0 - torch.sum(weights, dim=-2))

        return rgb

    def forward(
        self,
        rgb: TensorType[..., "num_samples", 3],
        weights: TensorType[..., "num_samples", 1],
    ) -> TensorType[..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample

        Returns:
            Outputs of rgb values.
        """

        rgb = self.combine_rgb(rgb, weights, background_color=self.background_color)
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb


class SHRenderer(nn.Module):
    """Render RGB value from spherical harmonics.

    Args:
        background_color: Background color as RGB. Defaults to random.
        activation: Output activation. Defaults to Sigmoid().
    """

    def __init__(
        self, background_color: Optional[TensorType[3]] = None, activation: Optional[nn.Module] = nn.Sigmoid()
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.activation = activation

    def forward(
        self,
        sh: TensorType[..., "num_samples", "coeffs"],
        directions: TensorType[..., "num_samples", 3],
        weights: TensorType[..., "num_samples", 1],
    ) -> TensorType[..., 3]:
        """Composite samples along ray and render color image

        Args:
            sh: Spherical hamonics coefficients for each sample
            directions: Sample direction
            weights: Weights for each sample

        Returns:
            Outputs of rgb values.
        """

        sh = sh.view(*sh.shape[:-1], 3, sh.shape[-1] // 3)

        levels = int(math.sqrt(sh.shape[-1]))
        components = components_from_spherical_harmonics(levels=levels, directions=directions)

        rgb = sh * components[..., None, :]  # [..., num_samples, 3, sh_components]
        rgb = torch.sum(sh, dim=-1) + 0.5  # [..., num_samples, 3]

        if self.activation is not None:
            self.activation(rgb)

        rgb = RGBRenderer.combine_rgb(rgb, weights, background_color=self.background_color)

        return rgb


class AccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: TensorType[..., "num_samples", 1],
    ) -> TensorType:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights: Weights for each sample

        Returns:
            Outputs of accumulated values.
        """

        accumulation = torch.sum(weights, dim=-2)
        return accumulation


class DepthRenderer(nn.Module):
    """Calculate depth along ray.

    Args:
        method (str, optional): Depth calculation method. Defaults to 'expected'.
    """

    def __init__(self, method: str = "expected") -> None:
        super().__init__()
        if method not in {"expected"}:
            raise ValueError(f"{method} is an invalid depth calculation method")
        self.method = method

    def forward(self, weights: TensorType[..., "num_samples", 1], ray_samples: RaySamples) -> TensorType[..., 1]:
        """Composite samples along ray and calculate disparities.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.

        Returns:
            Outputs of depth values.
        """

        if self.method == "expected":
            eps = 1e-10
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)

            depth = torch.clip(depth, steps[..., 0, :], steps[..., -1, :])

            return depth

        raise NotImplementedError(f"Method {self.method} not implemented")


class UncertaintyRenderer(nn.Module):
    """Calculate uncertainty along the ray."""

    @classmethod
    def forward(
        cls, betas: TensorType[..., "num_samples", 1], weights: TensorType[..., "num_samples", 1]
    ) -> TensorType:
        """Calculate uncertainty along the ray.

        Args:
            betas: Uncertainty betas for each sample.
            weights: Weights of each sample.

        Returns:
            Rendering of uncertainty.
        """
        uncertainty = torch.sum(weights * betas, dim=-2)
        return uncertainty


class SemanticRenderer(nn.Module):
    """Calculate semantics along the ray."""

    @classmethod
    def forward(
        cls, semantics: TensorType[..., "num_samples", "num_classes"], weights: TensorType[..., "num_samples", 1]
    ) -> TensorType[..., "num_classes"]:
        """_summary_"""
        sem = torch.sum(weights * semantics, dim=-2)
        return sem
