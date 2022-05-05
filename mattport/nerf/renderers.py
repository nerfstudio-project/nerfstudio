"""
Collection of renderers
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType


@dataclass
class RendererOutputs:
    """_summary_"""

    rgb: TensorType["num_rays", 3] = None
    density: TensorType["num_rays", 1] = None
    accumulation: TensorType["num_rays", 1] = None
    disparity: TensorType["num_rays", 1] = None


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


class DisparityRenderer(nn.Module):
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
            RendererOutputs: Outputs with disparity values.
        """

        if self.method == "expected":
            depth = torch.sum(weights * ts, dim=-1)
            eps = 1e-10
            disparity = 1.0 / torch.max(eps * torch.ones_like(depth), depth / torch.sum(weights, -1))

            return RendererOutputs(disparity=disparity)

        raise NotImplementedError(f"Method {self.method} not implemented")
