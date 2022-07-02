# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""Classic NeRF field"""


from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from pyrad.fields.modules.encoding import Encoding, Identity
from pyrad.fields.modules.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from pyrad.fields.modules.mlp import MLP
from pyrad.fields.base import Field
from pyrad.cameras.rays import RaySamples
from pyrad.fields.modules.spatial_distortions import SpatialDistortion


class NeRFField(Field):
    """NeRF Field

    Args:
        position_encoding (Encoding, optional): Position encoder. Defaults to Identity(in_dim=3).
        direction_encoding (Encoding, optional): Direction encoder. Defaults to Identity(in_dim=3).
        base_mlp_num_layers (int, optional): Number of layers for base MLP. Defaults to 8.
        base_mlp_layer_width (int, optional): Width of base MLP layers. Defaults to 256.
        head_mlp_num_layers (int, optional): Number of layer for ourput head MLP. Defaults to 2.
        head_mlp_layer_width (int, optional): Width of output head MLP layers. Defaults to 128.
        skip_connections (Tuple, optional): Where to add skip connection in base MLP. Defaults to (4,).
        use_integrated_encoding (bool, optional): Used integrated samples as encoding input, Defaults to False.
        spatial_distortion (SpatialDistortion, optional): Spatial distortion. Defaults to None.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())

    def get_density(self, ray_samples: RaySamples):
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
