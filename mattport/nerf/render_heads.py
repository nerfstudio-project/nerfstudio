"""
Collection of render heads
"""
from enum import Enum
from typing import Optional

from torch import nn
from torchtyping import TensorType


class RenderHeadKeys(Enum):
    """Keys for model outputs"""

    DENSITY = "density"
    DISTS = "dists"
    RGB = "rgb"
    SH_RGB = "sh_rgb"


class RenderHead(nn.Module):
    """Base Render head"""

    def __init__(self, in_dim: int, out_dim: int, activation: Optional[nn.Module]) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if activation:
            layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> TensorType[..., "out_dim"]:
        """Process network output for renderer

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Render head output
        """

        return self.net(in_tensor)


class DensityHead(RenderHead):
    """RGB head"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Softplus) -> None:
        super().__init__(in_dim, 1, activation)


class RGBHead(RenderHead):
    """RGB head"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Sigmoid) -> None:
        super().__init__(in_dim, 3, activation)
