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


from typing import Tuple

import torch
from torch.nn.parameter import Parameter

from pyrad.fields.modules.encoding import Encoding, HashEncoding, SHEncoding
from pyrad.fields.modules.field_heads import FieldHeadNames
from pyrad.fields.base import Field
from pyrad.fields.nerf_field import NeRFField
from pyrad.cameras.rays import RaySamples
from pyrad.utils.activations import trunc_exp

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_positions(positions, aabb):
    """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box."""
    aabb_lengths = aabb[1] - aabb[0]
    positions = (positions - aabb[0]) / aabb_lengths
    return positions


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]"""
    return (directions + 1.0) / 2.0


class TCNNInstantNGPField(Field):
    """NeRF Field"""

    def __init__(
        self, aabb, num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)

        self.geo_feat_dim = geo_feat_dim

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
        """Computes and returns the densities."""
        positions = get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding=None):
        # TODO: add valid_mask masking!
        # tcnn requires directions in the range [0,1]
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1)
        return {FieldHeadNames.RGB: rgb}


class TorchInstantNGPField(NeRFField):
    """
    PyTorch implementation of the instant-ngp field.
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

    def get_density(self, ray_samples: RaySamples):
        normalized_ray_samples = ray_samples
        normalized_ray_samples.positions = get_normalized_positions(
            normalized_ray_samples.frustums.get_positions(), self.aabb
        )
        return super().get_density(normalized_ray_samples)


field_implementation_to_class = {"tcnn": TCNNInstantNGPField, "torch": TorchInstantNGPField}
