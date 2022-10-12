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

"""TensoRF Field"""


from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


class TensoRFField(Field):
    """TensoRF Field

    Args:
        density_encoding:
        color_encoding:

        head_mlp_num_layers: Number of layer for ourput head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
    """

    def __init__(
        self,
        aabb,
        feature_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        density_encoding: Encoding = Identity(in_dim=3),
        color_encoding: Encoding = Identity(in_dim=3),
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.density_encoding = density_encoding
        self.color_encoding = color_encoding

        self.mlp_head = MLP(
            in_dim=27 + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
        )

        self.B = nn.Linear(in_features=self.color_encoding.get_out_dim(), out_features=27, bias=False)

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())

    def get_density(self, ray_samples: RaySamples):
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        density = self.density_encoding(positions)
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc, None

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        rgb_features = self.color_encoding(positions)
        rgb_features = self.B(rgb_features)

        d_encoded = self.direction_encoding(d)
        rgb_features_encoded = self.feature_encoding(rgb_features)

        mlp_out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
        rgb = self.field_output_rgb(mlp_out)
        return {FieldHeadNames.RGB: rgb}
