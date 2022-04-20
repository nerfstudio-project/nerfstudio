"""
Collection of renderers
"""
from dataclasses import dataclass
from typing import Set

import torch
from torch import nn
from torchtyping import TensorType

from mattport.nerf.field_modules import field_heads


@dataclass
class RendererOutputs:
    """_summary_"""

    rgb: TensorType["num_rays", 3] = None
    density: TensorType["num_rays", 1] = None


class Renderer(nn.Module):
    """Base Renderer. Intended to be subclassed"""

    def required_field_outputs(self) -> Set[field_heads.FieldHead]:
        """
        Returns:
            Set[RenderHead]: RenderHeads required for this renderer.
        """
        raise NotImplementedError

    def forward(
        self,
        rgb: TensorType[..., "num_samples", 3],
        density: TensorType[..., "num_samples", 1],
        deltas: TensorType[..., "num_samples"],
    ) -> RendererOutputs:
        """Composite samples along ray and render image

        Args:
            in_tensor (TensorType[..., "num_samples", -1]): feature for each sample
            deltas (TensorType[..., "num_samples"]): depth of each sample

        Returns:
            TensorType[..., "out_dim"]: Rendered ray
        """
        raise NotImplementedError


class RGBRenderer(Renderer):
    """Standard volumetic rendering."""

    def required_field_outputs(self) -> Set[field_heads.FieldHead]:
        return set(field_heads.RGBFieldHead, field_heads.DensityFieldHead)

    def forward(
        self,
        rgb: TensorType[..., "num_samples", 3],
        density: TensorType[..., "num_samples", 1],
        deltas: TensorType[..., "num_samples"],
    ) -> RendererOutputs:
        """Composite samples along ray and render color image

        Args:
            in_tensor (TensorType[..., "num_samples", -1]): RGB for each sample
            deltas (TensorType[..., "num_samples"]): depth influence for each sample

        Returns:
            TensorType[..., "out_dim"]: Composited RGB ray
        """
        delta_density = deltas * density[..., 0]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1], dim=-1)
        transmittance = torch.cat([torch.zeros((*transmittance.shape[:1], 1)).to(rgb.device), transmittance], axis=-1)
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        rgb = torch.sum(weights[..., None] * rgb, dim=-2)

        renderer_outputs = RendererOutputs(rgb=rgb)
        return renderer_outputs


class DepthRenderer(Renderer):
    """_summary_

    Args:
        Renderer (_type_): _description_
    """

    def required_field_outputs(self) -> Set[field_heads.FieldHead]:
        return set(field_heads.DensityFieldHead)

    def forward(
        self,
        rgb: TensorType[..., "num_samples", 3],
        density: TensorType[..., "num_samples", 1],
        deltas: TensorType[..., "num_samples"],
    ) -> RendererOutputs:
        raise NotImplementedError
