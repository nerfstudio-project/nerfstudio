"""
Collection of render heads
"""
from dataclasses import dataclass
from typing import Optional

from torch import nn
from torchtyping import TensorType

from mattport.nerf.field_modules.base import FieldModule


@dataclass
class FieldHeadOutputs:
    """_summary_"""

    rgb: TensorType["...", 3] = None
    density: TensorType["...", 1] = None


class FieldHead(FieldModule):
    """Base field output"""

    def __init__(self, in_dim: int, out_dim: int, activation: Optional[nn.Module] = None) -> None:
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension for renderer
            activation (Optional[nn.Module]): output head activation
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        layers = [nn.Linear(self.in_dim, self.out_dim)]
        if self.activation:
            layers.append(self.activation)
        self.net = nn.Sequential(*layers)

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> TensorType[..., "out_dim"]:
        """Process network output for renderer

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Render head output
        """
        if not self.field_quantity_name:
            raise ValueError("field_quantity_name should be set in the child class. E.g., as 'rgb' or 'density'.")
        if not self.net:
            raise SystemError("Render head network not initialized. build_nn_modules() should be called.")
        out_tensor = self.net(in_tensor)
        field_head_outputs = FieldHeadOutputs()
        setattr(field_head_outputs, self.field_quantity_name, out_tensor)
        return field_head_outputs

class DensityFieldHead(FieldHead):
    """Density output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim, 1, activation)
        self.field_quantity_name = "density"


class RGBFieldHead(FieldHead):
    """RGB output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim, 3, activation)
        self.field_quantity_name = "rgb"
