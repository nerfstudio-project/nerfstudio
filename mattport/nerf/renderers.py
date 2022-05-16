"""
Collection of renderers
"""
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType

from mattport.utils.math import components_from_spherical_harmonics


@dataclass
class RendererOutputs:
    """_summary_"""

    rgb: TensorType["num_rays", 3] = None
    density: TensorType["num_rays", 1] = None
    accumulation: TensorType["num_rays", 1] = None
    depth: TensorType["num_rays", 1] = None


class RGBRenderer(nn.Module):
    """Standard volumetic rendering."""

    def __init__(self, background_color: Optional[TensorType[3]] = None) -> None:
        """
        Args:
            background_color (TensorType[3], optional): Background color as RGB. Defaults to black.
        """
        super().__init__()
        self.background_color = background_color

    def forward(
        self,
        rgb: TensorType[..., "num_samples", 3],
        weights: TensorType[..., "num_samples"],
    ) -> RendererOutputs:
        """Composite samples along ray and render color image

        Args:
            rgb (TensorType[..., "num_samples", -1]): RGB for each sample
            weights (TensorType[..., "num_samples"]): Weights for each sample

        Returns:
            RendererOutputs: Outputs with rgb values.
        """

        rgb = torch.sum(weights[..., None] * rgb, dim=-2)

        if self.background_color is not None:
            rgb = rgb + self.background_color.to(weights.device)[None, ...] * (
                1.0 - torch.sum(weights, dim=-1)[..., None]
            )
        torch.clamp_(rgb, min=0.0, max=1.0)

        assert torch.max(rgb) <= 1.0
        assert torch.min(rgb) >= 0.0

        renderer_outputs = RendererOutputs(rgb=rgb)
        return renderer_outputs


class SHRenderer(nn.Module):
    """Render RGB value from spherical harmonics."""

    def __init__(
        self, activation: Optional[nn.Module] = nn.Sigmoid(), background_color: Optional[TensorType[3]] = None
    ) -> None:
        """
        Args:
            background_color (TensorType[3], optional): Background color as RGB. Defaults to black.
            activation (Optional[nn.Module], optional): Output activation. Defaults to Sigmoid().
        """
        super().__init__()
        self.background_color = background_color
        self.activation = activation

    def forward(
        self,
        sh: TensorType[..., "num_samples", "coeffs"],
        directions: TensorType[..., "num_samples", 3],
        weights: TensorType[..., "num_samples"],
    ) -> RendererOutputs:
        """Composite samples along ray and render color image

        Args:
            sh (TensorType[..., "num_samples", "coeffs"]): Spherical hamonics coefficients for each sample
            directions: (TensorType[..., "num_samples", 3]): Sample direction
            weights (TensorType[..., "num_samples"]): Weights for each sample

        Returns:
            RendererOutputs: Outputs with rgb values.
        """

        sh = sh.view(*sh.shape[:-1], 3, sh.shape[-1] // 3)

        levels = int(math.sqrt(sh.shape[-1]))
        components = components_from_spherical_harmonics(levels=levels, directions=directions)

        rgb = sh * components[..., None, :]  # [..., num_samples, 3, sh_components]
        rgb = torch.sum(sh, dim=-1)  # [..., num_samples, 3]

        rgb = torch.sum(weights[..., None] * rgb, dim=-2) + 0.5

        if self.activation is not None:
            self.activation(rgb)

        if self.background_color is not None:
            rgb = rgb + self.background_color.to(weights.device)[None, ...] * (
                1.0 - torch.sum(weights, dim=-1)[..., None]
            )
        torch.clamp_(rgb, min=0.0, max=1.0)

        assert torch.max(rgb) <= 1.0
        assert torch.min(rgb) >= 0.0

        renderer_outputs = RendererOutputs(rgb=rgb)
        return renderer_outputs


class AccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: TensorType[..., "num_samples"],
    ) -> RendererOutputs:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights (TensorType[..., "num_samples"]): Weights for each sample

        Returns:
            RendererOutputs: Outputs with accumulated values.
        """

        accumulation = torch.sum(weights, dim=-1)[..., None]

        renderer_outputs = RendererOutputs(accumulation=accumulation)
        return renderer_outputs


class DepthRenderer(nn.Module):
    """Calcualte depth along ray."""

    def __init__(self, method: str = "expected") -> None:
        """
        Args:
            method (str, optional): Depth calculation method. Defaults to 'expected'.
        """
        super().__init__()
        if method not in {"expected"}:
            raise ValueError(f"{method} is an invalid depth calculation method")
        self.method = method

    def forward(self, weights: TensorType[..., "num_samples"], ts: TensorType[..., "num_samples"]) -> RendererOutputs:
        """Composite samples along ray and calculate disparities.

        Args:
            weights (TensorType[..., "num_samples"]): Weights for each sample
            ts (TensorType[..., "num_samples"]): Sample locations along rays

        Returns:
            RendererOutputs: Outputs with depth values.
        """

        if self.method == "expected":
            eps = 1e-10
            depth = torch.sum(weights * ts, dim=-1) / (torch.sum(weights, -1) + eps)

            depth = torch.clip(depth, ts[..., 0], ts[..., -1])

            return RendererOutputs(depth=depth[..., None])

        raise NotImplementedError(f"Method {self.method} not implemented")
