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
from nerfstudio.field_components.encodings import Encoding, Identity, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field


class TensoRFField(Field):
    """TensoRF Field"""

    def __init__(
        self,
        aabb: TensorType,
        # the aabb bounding box of the dataset
        feature_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for appearance encoding outputs
        direction_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for ray direction
        density_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene density
        color_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene color
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        head_mlp_num_layers: int = 2,
        # number of layers for the MLP
        head_mlp_layer_width: int = 128,
        # layer width for the MLP
        use_sh: bool = False,
        # whether to use spherical harmonics as the feature decoding function
        sh_levels: int = 2,
        # number of levels to use for spherical harmonics
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.density_encoding = density_encoding
        self.color_encoding = color_encoding

        self.mlp_head = MLP(
            in_dim=appearance_dim + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.use_sh = use_sh

        if self.use_sh:
            self.sh = SHEncoding(sh_levels)
            self.B = nn.Linear(
                in_features=self.color_encoding.get_out_dim(), out_features=3 * self.sh.get_out_dim(), bias=False
            )
        else:
            self.B = nn.Linear(in_features=self.color_encoding.get_out_dim(), out_features=appearance_dim, bias=False)

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())

    def get_density(self, ray_samples: RaySamples) -> TensorType:
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        density = self.density_encoding(positions)
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = torch.nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> TensorType:
        d = ray_samples.frustums.directions
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = positions * 2 - 1
        rgb_features = self.color_encoding(positions)
        rgb_features = self.B(rgb_features)

        d_encoded = self.direction_encoding(d)
        rgb_features_encoded = self.feature_encoding(rgb_features)

        if self.use_sh:
            sh_mult = self.sh(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
            rgb = self.field_output_rgb(out)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[TensorType] = None,
        bg_color: Optional[TensorType] = None,
    ) -> Dict[FieldHeadNames, TensorType]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with TensoRF")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density = self.get_density(input_rays)
                rgb = self.get_outputs(input_rays, None)

                base_density[mask] = density
                base_rgb[mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()

            density = base_density
            rgb = base_rgb
        else:
            density = self.get_density(ray_samples)
            rgb = self.get_outputs(ray_samples, None)

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
