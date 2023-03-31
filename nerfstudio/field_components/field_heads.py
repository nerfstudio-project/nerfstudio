# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Collection of render heads
"""
from enum import Enum
from typing import Callable, Optional, Union

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.base_field_component import FieldComponent


class FieldHeadNames(Enum):
    """Possible field outputs"""

    RGB = "rgb"
    SH = "sh"
    DENSITY = "density"
    NORMALS = "normals"
    PRED_NORMALS = "pred_normals"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS = "semantics"
    SDF = "sdf"
    ALPHA = "alpha"
    GRADIENT = "gradient"


class FieldHead(FieldComponent):
    """Base field output

    Args:
        out_dim: output dimension for renderer
        field_head_name: Field type
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(
        self,
        out_dim: int,
        field_head_name: FieldHeadNames,
        in_dim: Optional[int] = None,
        activation: Optional[Union[nn.Module, Callable]] = None,
    ) -> None:

        super().__init__()
        self.out_dim = out_dim
        self.activation = activation
        self.field_head_name = field_head_name
        self.net = None
        if in_dim is not None:
            self.in_dim = in_dim
            self._construct_net()

    def set_in_dim(self, in_dim: int) -> None:
        """Set input dimension of Field Head"""
        self.in_dim = in_dim
        self._construct_net()

    def _construct_net(self):
        self.net = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process network output for renderer

        Args:
            in_tensor: Network input

        Returns:
            Render head output
        """
        if not self.net:
            raise SystemError("in_dim not set. Must be provided to constructor, or set_in_dim() should be called.")
        out_tensor = self.net(in_tensor)
        if self.activation:
            out_tensor = self.activation(out_tensor)
        return out_tensor


class DensityFieldHead(FieldHead):
    """Density output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, field_head_name=FieldHeadNames.DENSITY, activation=activation)


class RGBFieldHead(FieldHead):
    """RGB output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.RGB, activation=activation)


class SHFieldHead(FieldHead):
    """Spherical harmonics output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        levels: Number of spherical harmonics layers.
        channels: Number of channels. Defaults to 3 (ie RGB).
        activation: Output activation.
    """

    def __init__(
        self, in_dim: Optional[int] = None, levels: int = 3, channels: int = 3, activation: Optional[nn.Module] = None
    ) -> None:

        out_dim = channels * levels**2
        super().__init__(in_dim=in_dim, out_dim=out_dim, field_head_name=FieldHeadNames.SH, activation=activation)


class UncertaintyFieldHead(FieldHead):
    """Uncertainty output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(in_dim=in_dim, out_dim=1, field_head_name=FieldHeadNames.UNCERTAINTY, activation=activation)


class TransientRGBFieldHead(FieldHead):
    """Transient RGB output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Sigmoid()) -> None:
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.TRANSIENT_RGB, activation=activation)


class TransientDensityFieldHead(FieldHead):
    """Transient density output

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Softplus()) -> None:
        super().__init__(
            in_dim=in_dim, out_dim=1, field_head_name=FieldHeadNames.TRANSIENT_DENSITY, activation=activation
        )


class SemanticFieldHead(FieldHead):
    """Semantic output

    Args:
        num_classes: Number of semantic classes
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, num_classes: int, in_dim: Optional[int] = None) -> None:
        super().__init__(in_dim=in_dim, out_dim=num_classes, field_head_name=FieldHeadNames.SEMANTICS, activation=None)


class PredNormalsFieldHead(FieldHead):
    """Predicted normals output.

    Args:
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, in_dim: Optional[int] = None, activation: Optional[nn.Module] = nn.Tanh()) -> None:
        super().__init__(in_dim=in_dim, out_dim=3, field_head_name=FieldHeadNames.PRED_NORMALS, activation=activation)

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Needed to normalize the output into valid normals."""
        out_tensor = super().forward(in_tensor)
        out_tensor = torch.nn.functional.normalize(out_tensor, dim=-1)
        return out_tensor
