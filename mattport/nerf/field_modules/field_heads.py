"""
Collection of render heads
"""
from enum import Enum
from typing import Dict, Optional

from torch import nn
from torchtyping import TensorType

from mattport.nerf.field_modules.base import FieldModule


class FieldHeadNames(Enum):
    """Possible field outputs"""

    RGB = "rgb"
    DENSITY = "density"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS_STUFF = "semantics_stuff"
    SEMANTICS_THING = "semantics_thing"


class FieldHead(FieldModule):
    """Base field output"""

    def __init__(
        self, in_dim: int, out_dim: int, field_head_name: FieldHeadNames, activation: Optional[nn.Module] = None
    ) -> None:
        """
        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension for renderer
            field_head_name (FieldHeadNames): name of field head
            activation (Optional[nn.Module]): output head activation
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # if field_head_name not in FieldHeadNames:
        #     raise ValueError("Incorrect field output name. Should be one of FieldHeadNames.")
        self.field_head_name = field_head_name
        self.activation = activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        layers = [nn.Linear(self.in_dim, self.out_dim)]
        if self.activation:
            layers.append(self.activation)
        self.net = nn.Sequential(*layers)

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> Dict[FieldHeadNames, TensorType]:
        """Process network output for renderer

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Render head output
        """
        if not self.field_head_name:
            raise ValueError("field_quantity_name should be set in the child class. E.g., as 'rgb' or 'density'.")
        if not self.net:
            raise SystemError("Render head network not initialized. build_nn_modules() should be called.")
        out_tensor = self.net(in_tensor)
        return {self.field_head_name: out_tensor}


class DensityFieldHead(FieldHead):
    """Density output"""

    def __init__(
        self, in_dim: int, field_head_name=FieldHeadNames.DENSITY, activation: Optional[nn.Module] = nn.Softplus()
    ) -> None:
        super().__init__(in_dim, out_dim=1, field_head_name=field_head_name, activation=activation)


class RGBFieldHead(FieldHead):
    """RGB output"""

    def __init__(
        self, in_dim: int, field_head_name=FieldHeadNames.RGB, activation: Optional[nn.Module] = nn.Sigmoid()
    ) -> None:
        super().__init__(in_dim, out_dim=3, field_head_name=field_head_name, activation=activation)


class UncertaintyFieldHead(FieldHead):
    """Uncertainty output"""

    def __init__(
        self, in_dim: int, field_head_name=FieldHeadNames.UNCERTAINTY, activation: Optional[nn.Module] = nn.Softplus()
    ) -> None:
        super().__init__(in_dim, out_dim=1, field_head_name=field_head_name, activation=activation)


class TransientRGBHead(RGBFieldHead):
    """Transient RGB output"""

    def __init__(
        self, in_dim: int, field_head_name=FieldHeadNames.TRANSIENT_RGB, activation: Optional[nn.Module] = nn.Sigmoid()
    ) -> None:
        super().__init__(in_dim, field_head_name, activation)


class TransientDensityHead(DensityFieldHead):
    """Transient density output"""

    def __init__(
        self,
        in_dim: int,
        field_head_name=FieldHeadNames.TRANSIENT_DENSITY,
        activation: Optional[nn.Module] = nn.Softplus(),
    ) -> None:
        super().__init__(in_dim, field_head_name, activation)


class SemanticStuffHead(FieldHead):
    """Semantic stuff output"""

    def __init__(self, in_dim: int, num_classes: int, field_head_name=FieldHeadNames.SEMANTICS_STUFF) -> None:
        super().__init__(in_dim, out_dim=num_classes, field_head_name=field_head_name, activation=None)
