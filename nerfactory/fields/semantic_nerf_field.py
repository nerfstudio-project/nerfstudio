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

"""
Semantic NeRF field implementation.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfactory.cameras.rays import RaySamples
from nerfactory.fields.base import Field
from nerfactory.fields.modules.encoding import Encoding, Identity
from nerfactory.fields.modules.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
    SemanticFieldHead,
)
from nerfactory.fields.modules.mlp import MLP


class SemanticNerfField(Field):
    """Semantic-NeRF field"""

    def __init__(
        self,
        num_semantic_classes: int,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
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
        self.mlp_semantic = MLP(
            in_dim=self.mlp_head.get_out_dim(),
            layer_width=self.mlp_head.layer_width // 2,
            num_layers=1,
            activation=nn.ReLU(),
        )
        self.field_head_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_head_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())
        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(), num_classes=self.num_semantic_classes
        )

    def get_density(self, ray_samples: RaySamples):
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_head_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
        outputs = {}
        # rgb
        outputs[self.field_head_rgb.field_head_name] = self.field_head_rgb(mlp_out)
        # semantic
        mlp_out_sem = self.mlp_semantic(mlp_out)
        outputs[self.field_head_semantic.field_head_name] = self.field_head_semantic(mlp_out_sem)
        return outputs
