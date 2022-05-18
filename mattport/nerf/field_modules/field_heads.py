"""
Collection of render heads
"""
from enum import Enum
from typing import Optional

from torch import nn
from torchtyping import TensorType

from mattport.nerf.field_modules.base import FieldModule


class FieldHeadNames(Enum):
    """Possible field outputs"""

    RGB = "rgb"
    SH = "sh"
    DENSITY = "density"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS_STUFF = "semantics_stuff"
    SEMANTICS_THING = "semantics_thing"


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
        layers = [nn.Linear(self.in_dim, self.out_dim)]
        if self.activation:
            layers.append(self.activation)
        self.net = nn.Sequential(*layers)

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> TensorType:
        """Process network output for renderer

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Render head output
        """
        if not self.net:
            raise SystemError("Render head network not initialized. build_nn_modules() should be called.")
        out_tensor = self.net(in_tensor)
        return out_tensor


class DensityFieldHead(FieldHead):
    """Density output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim, out_dim=1, activation=activation)


class RGBFieldHead(FieldHead):
    """RGB output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim, out_dim=3, activation=activation)


class SHFieldHead(FieldHead):
    """Spherical harmonics output"""

    def __init__(self, in_dim: int, levels: int = 3, channels: int = 3, activation: Optional[nn.Module] = None) -> None:
        """_summary_

        Args:
            in_dim (int): Input dimension
            levels (int, optional): Number of spherical harmonics layers. Defaults to 3.
            channels (int, optional): Number of channels. Defaults to 3 (ie RGB).
            activation (Optional[nn.Module], optional): Output activation. Defaults to None.
        """
        out_dim = channels * levels**2
        super().__init__(in_dim, out_dim=out_dim, activation=activation)


class UncertaintyFieldHead(FieldHead):
    """Uncertainty output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim, out_dim=1, activation=activation)


class TransientRGBHead(FieldHead):
    """Transient RGB output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim, out_dim=3, activation=activation)


class TransientDensityHead(FieldHead):
    """Transient density output"""

    def __init__(self, in_dim: int, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim, out_dim=1, activation=activation)


class SemanticStuffHead(FieldHead):
    """Semantic stuff output"""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        """
        Args:
            in_dim (int): Input dimension
            num_classes (int): Number of semantic classes
        """
        super().__init__(in_dim, out_dim=num_classes, activation=None)
