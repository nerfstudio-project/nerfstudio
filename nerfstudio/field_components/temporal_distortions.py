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

"""Space distortions which occur as a function of time."""

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.encodings import Encoding, NeRFEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.utils.math import hutchinson_div_approx


class TemporalDistortion(nn.Module):
    """Apply spatial distortions as a function of time"""

    def forward(self, positions: TensorType["bs":..., 3], times: Optional[TensorType[1]]) -> TensorType["bs":..., 3]:
        """
        Args:
            positions: Samples to translate as a function of time
            times: times for each sample

        Returns:
            Translated positions.
        """

    @abstractmethod
    def get_loss_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """
        Returns:
          Dict of loss names to relevant losses. May be empty.
        """
        return {}


class TemporalDistortionKind(Enum):
    """Possible temporal distortion names"""

    DNERF = "dnerf"
    NRNERF = "nrnerf"

    def to_temporal_distortion(self, config: Dict[str, Any]) -> TemporalDistortion:
        """Converts this kind to a temporal distortion"""
        if self == TemporalDistortionKind.DNERF:
            return DNeRFDistortion(**config)
        if self == TemporalDistortionKind.NRNERF:
            return NRNeRFDistortion(**config)
        raise NotImplementedError(f"Unknown temporal distortion kind {self}")


class DNeRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        position_encoding: Encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        temporal_encoding: Encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.mlp_deform = MLP(
            in_dim=self.position_encoding.get_out_dim() + self.temporal_encoding.get_out_dim(),
            out_dim=3,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )

    def get_loss_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, positions, times=None):
        if times is None:
            return None
        p = self.position_encoding(positions)
        t = self.temporal_encoding(times)
        return self.mlp_deform(torch.cat([p, t], dim=-1))


class NRNeRFDistortion(TemporalDistortion):
    """Optimizable temporal deformation using an MLP.
    Args:
        position_encoding: An encoding for the XYZ of distortion
        temporal_encoding: An encoding for the time of distortion
        mlp_num_layers: Number of layers in distortion MLP
        mlp_layer_width: Size of hidden layer for the MLP
        skip_connections: Number of layers for skip connections in the MLP
    """

    def __init__(
        self,
        position_encoding: Encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        temporal_encoding: Encoding = NeRFEncoding(
            in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ),
        mlp_num_layers: int = 4,
        mlp_layer_width: int = 256,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.temporal_encoding = temporal_encoding
        self.mlp_deform = MLP(
            in_dim=self.position_encoding.get_out_dim() + self.temporal_encoding.get_out_dim(),
            out_dim=4,
            num_layers=mlp_num_layers,
            layer_width=mlp_layer_width,
            skip_connections=skip_connections,
        )
        self.rigidity = None
        self.p = None
        self.raw_offset = None
        self.rigid_offset = None

    def forward(self, positions, times=None):
        if times is None:
            return None
        self.p = positions.requires_grad_()
        p = self.position_encoding(self.p)
        t = self.temporal_encoding(times)
        self.raw_offset, self.rigidity = self.mlp_deform(torch.cat([p, t], dim=-1)).split([3, 1], dim=-1)
        self.rigidity = self.rigidity.sigmoid()

        self.rigid_offset = self.raw_offset * self.rigidity
        return self.rigid_offset

    def get_loss_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        if self.p is None:
            return {}
        div_loss = hutchinson_div_approx(self.p, self.rigid_offset).square()
        # div_coarse = (div_loss * outputs["alphas_coarse"]).mean()
        div_fine = (div_loss * outputs["alphas_fine"].squeeze(-1)).mean()

        norm_dp = torch.linalg.vector_norm(self.raw_offset, dim=-1, keepdim=True)
        norm_dp = norm_dp.pow(2 - self.rigidity)

        # offset_coarse = outputs["weights_coarse"] * (norm_dp + 3e-3 * self.rigidity)
        offset_fine = (outputs["weights_fine"] * (norm_dp + 3e-3 * self.rigidity)).mean()

        return {
            # "divergence_coarse": div_coarse,
            "divergence_fine": div_fine,
            # "offset_coarse": offset_coarse,
            "offset_fine": offset_fine,
        }
