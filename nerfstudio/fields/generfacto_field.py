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
Field for Generfacto model
"""


from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field, get_normalized_directions


class GenerfactoField(Field):
    """Generfacto Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
    """

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 256,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim

        base_res = 16
        features_per_level = 2
        np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        encoder = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        network = MLP(
            in_dim=encoder.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_base = torch.nn.Sequential(encoder, network)

        self.mlp_background_color = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=2,
            layer_width=32,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.mlp_head = MLP(
            in_dim=self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_background_rgb(self, ray_bundle: RayBundle) -> Tensor:
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)

        outputs_shape = ray_bundle.directions.shape[:-1]
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        background_rgb = self.mlp_background_color(directions_flat).view(*outputs_shape, -1).to(directions)

        return background_rgb

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}

        directions = get_normalized_directions(ray_samples.frustums.directions)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        h = density_embedding.view(-1, self.geo_feat_dim)

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
