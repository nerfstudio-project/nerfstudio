"""
Collection of renderers
"""
from typing import Dict, Set

import torch
from torch import nn
from torchtyping import TensorType

from mattport.nerf.field_modules import field_outputs


class Renderer(nn.Module):
    """Base Renderer. Intended to be subclassed"""

    def required_field_outputs(self) -> Set[field_outputs.FieldOutput]:
        """
        Returns:
            Set[RenderHead]: RenderHeads required for this renderer.
        """
        raise NotImplementedError

    def forward(
        self,
        field_output_dict: Dict["str", TensorType[..., "num_samples", -1]],
        deltas: TensorType[..., "num_samples"],
    ) -> TensorType[..., "out_dim"]:
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

    def __init__(self) -> None:
        super().__init__()

    def required_field_outputs(self) -> Set[field_outputs.FieldOutput]:
        return set(field_outputs.RGBFieldOutput, field_outputs.DensityFieldOutput)

    def forward(
        self,
        rgb: TensorType[..., "num_samples", 3],
        density: TensorType[..., "num_samples", 1],
        deltas: TensorType[..., "num_samples"],
    ) -> TensorType[..., "out_dim"]:
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
        transmittance = torch.cat([torch.zeros((*transmittance.shape[:1], 1)), transmittance], axis=-1)
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        rgb = torch.sum(weights[..., None] * rgb, dim=-2)

        return rgb


class DepthRenderer(Renderer):
    """_summary_

    Args:
        Renderer (_type_): _description_
    """

    def required_field_outputs(self) -> Set[field_outputs.FieldOutput]:
        return set(field_outputs.DensityFieldOutput)

    def forward(
        self,
        field_output_dict: Dict["str", TensorType[..., "num_samples", -1]],
        deltas: TensorType[..., "num_samples"],
    ) -> TensorType[..., "out_dim"]:
        raise NotImplementedError
