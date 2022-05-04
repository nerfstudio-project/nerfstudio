"""
Collection of renderers
"""
from dataclasses import dataclass

import torch
from torch import nn
from torchtyping import TensorType


@dataclass
class RendererOutputs:
    """_summary_"""

    rgb: TensorType["num_rays", 3] = None
    density: TensorType["num_rays", 1] = None


class RGBRenderer(nn.Module):
    """Standard volumetic rendering."""

    def __init__(self, white_background: bool = False) -> None:
        """
        Args:
            white_background (bool, optional): Composite onto white. Defaults to True.
        """
        super().__init__()
        self.white_background = white_background

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
            TensorType[..., "out_dim"]: Composited RGB ray
        """

        rgb = torch.sum(weights[..., None] * rgb, dim=-2)

        if self.white_background:
            rgb = rgb + (1.0 - torch.sum(weights, dim=-1))[..., None]

        renderer_outputs = RendererOutputs(rgb=rgb)
        return renderer_outputs
