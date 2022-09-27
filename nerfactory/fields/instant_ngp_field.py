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
Instant-NGP field implementations using tiny-cuda-nn, torch, ....
"""


from typing import Optional, Tuple

import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfactory.cameras.rays import RaySamples
from nerfactory.datamanagers.structs import SceneBounds
from nerfactory.fields.base import Field
from nerfactory.fields.modules.encoding import Encoding, HashEncoding, SHEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.modules.spatial_distortions import SpatialDistortion
from nerfactory.fields.nerf_field import NeRFField
from nerfactory.utils.activations import trunc_exp

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


class TCNNInstantNGPField(Field):
    """TCNN implementation of the Instant-NGP field.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
    """

    def __init__(
        self,
        aabb,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion

        # TODO: set this properly based on the aabb
        per_level_scale = 1.4472692012786865

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim,
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
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBounds.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        # assert all positions are in the range [0, 1]
        # otherwise print min and max values
        # assert torch.all(positions >= 0.0) and torch.all(
        #     positions <= 1.0
        # ), f"positions: {positions.min()} {positions.max()}"
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        # TODO: add valid_mask masking!
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        # assert all directions are in the range [0, 1]
        # assert torch.all(directions >= 0.0) and torch.all(
        #     directions <= 1.0
        # ), f"directions: {directions.min()} {directions.max()}"
        d = self.direction_encoding(directions_flat)
        if density_embedding is None:
            positions = SceneBounds.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            h = torch.cat([d, positions.view(-1, 3)], dim=-1)
        else:
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        return {FieldHeadNames.RGB: rgb}


class TorchInstantNGPField(NeRFField):
    """
    PyTorch implementation of the Instant-NGP field.
    """

    def __init__(
        self,
        aabb,
        position_encoding: Encoding = HashEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 3,
        base_mlp_layer_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        skip_connections: Tuple = (4,),
    ) -> None:
        super().__init__(
            position_encoding,
            direction_encoding,
            base_mlp_num_layers,
            base_mlp_layer_width,
            head_mlp_num_layers,
            head_mlp_layer_width,
            skip_connections,
        )
        self.aabb = Parameter(aabb, requires_grad=False)


field_implementation_to_class = {"tcnn": TCNNInstantNGPField, "torch": TorchInstantNGPField}
