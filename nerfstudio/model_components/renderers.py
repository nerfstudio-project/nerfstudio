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
Collection of renderers

Example:

.. code-block:: python

    field_outputs = field(ray_sampler)
    weights = ray_sampler.get_weights(field_outputs[FieldHeadNames.DENSITY])

    rgb_renderer = RGBRenderer()
    rgb = rgb_renderer(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

"""
import contextlib
import math
from typing import Generator, Literal, Optional, Union

import nerfacc
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize

BACKGROUND_COLOR_OVERRIDE: Optional[TensorType[3]] = None


@contextlib.contextmanager
def background_color_override_context(mode: TensorType[3]) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE  # pylint: disable=global-statement
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color


class RGBRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color: Union[Literal["random", "last_sample"], TensorType[3]] = "random") -> None:
        super().__init__()
        self.background_color = background_color

    @classmethod
    def combine_rgb(
        cls,
        rgb: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        background_color: Union[Literal["random", "white", "black", "last_sample"], TensorType[3]] = "random",
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        if background_color == "random":
            background_color = torch.rand_like(comp_rgb).to(rgb.device)
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color].to(rgb.device)

        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color.to(weights.device) * (1.0 - accumulated_weight)

        return comp_rgb

    def forward(
        self,
        rgb: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of rgb values.
        """

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = self.combine_rgb(
            rgb, weights, background_color=self.background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb


class SHRenderer(nn.Module):
    """Render RGB value from spherical harmonics.

    Args:
        background_color: Background color as RGB. Uses random colors if None
        activation: Output activation.
    """

    def __init__(
        self,
        background_color: Union[Literal["random", "last_sample"], TensorType[3]] = "random",
        activation: Optional[nn.Module] = nn.Sigmoid(),
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
            sh: Spherical harmonics coefficients for each sample
            directions: Sample direction
            weights: Weights for each sample

        Returns:
            Outputs of rgb values.
        """

        sh = sh.view(*sh.shape[:-1], 3, sh.shape[-1] // 3)

        levels = int(math.sqrt(sh.shape[-1]))
        components = components_from_spherical_harmonics(levels=levels, directions=directions)

        rgb = sh * components[..., None, :]  # [..., num_samples, 3, sh_components]
        rgb = torch.sum(rgb, dim=-1)  # [..., num_samples, 3]

        if self.activation is not None:
            rgb = self.activation(rgb)

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = RGBRenderer.combine_rgb(rgb, weights, background_color=self.background_color)
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)

        return rgb


class AccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: TensorType["bs":..., "num_samples", 1],
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType["bs":..., 1]:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of accumulated values.
        """

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            accumulation = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            accumulation = torch.sum(weights, dim=-2)
        return accumulation


class DepthRenderer(nn.Module):
    """Calculate depth along ray.

    Depth Method:
        - median: Depth is set to the distance where the accumulated weight reaches 0.5.
        - expected: Expected depth along ray. Same procedure as rendering rgb, but with depth.

    Args:
        method: Depth calculation method.
    """

    def __init__(self, method: Literal["median", "expected"] = "median") -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        weights: TensorType[..., "num_samples", 1],
        ray_samples: RaySamples,
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType[..., 1]:
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of depth values.
        """

        if self.method == "median":
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError("Median depth calculation is not implemented for packed samples.")
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
            split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
            median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
            median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]
            return median_depth
        if self.method == "expected":
            eps = 1e-10
            steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                # Necessary for packed samples from volumetric ray sampler
                depth = nerfacc.accumulate_along_rays(
                    weights[..., 0], values=steps, ray_indices=ray_indices, n_rays=num_rays
                )
                accumulation = nerfacc.accumulate_along_rays(
                    weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
                )
                depth = depth / (accumulation + eps)
            else:
                depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)

            depth = torch.clip(depth, steps.min(), steps.max())

            return depth

        raise NotImplementedError(f"Method {self.method} not implemented")


class UncertaintyRenderer(nn.Module):
    """Calculate uncertainty along the ray."""

    @classmethod
    def forward(
        cls, betas: TensorType["bs":..., "num_samples", 1], weights: TensorType["bs":..., "num_samples", 1]
    ) -> TensorType["bs":..., 1]:
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
        cls,
        semantics: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        sem = torch.sum(weights * semantics, dim=-2)
        return sem


class NormalsRenderer(nn.Module):
    """Calculate normals along the ray."""

    @classmethod
    def forward(
        cls,
        normals: TensorType["bs":..., "num_samples", 3],
        weights: TensorType["bs":..., "num_samples", 1],
        normalize: bool = True,
    ) -> TensorType["bs":..., 3]:
        """Calculate normals along the ray.

        Args:
            normals: Normals for each sample.
            weights: Weights of each sample.
            normalize: Normalize normals.
        """
        n = torch.sum(weights * normals, dim=-2)
        if normalize:
            n = safe_normalize(n)
        return n
