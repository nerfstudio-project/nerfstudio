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
Field for dreamfusion model
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


class DreamFusionField(Field):
    """DreamFusion Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion

        num_levels = 16
        max_res = 1024
        base_res = 16
        log2_hashmap_size = 19
        features_per_level = 2
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # predicted normals
        self.mlp_pred_normals = tcnn.Network(
            n_input_dims=self.geo_feat_dim + self.position_encoding.n_output_dims,
            n_output_dims=hidden_dim_transient,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.n_output_dims)

        self.mlp_background_color = tcnn.Network(
            n_input_dims=self.position_encoding.n_output_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
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

    def get_background_rgb(self, ray_bundle: RayBundle):
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)

        outputs_shape = ray_bundle.directions.shape[:-1]

        positions_flat = self.position_encoding(directions.view(-1, 3))
        background_rgb = self.mlp_background_color(positions_flat).view(*outputs_shape, -1).to(directions)
        return background_rgb


    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        assert density_embedding is not None
        outputs = {}

        directions = get_normalized_directions(ray_samples.frustums.directions)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # predicted normals
        positions = ray_samples.frustums.get_positions()

        positions_flat = self.position_encoding(positions.view(-1, 3))
        pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = density_embedding.view(-1, self.geo_feat_dim)

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
