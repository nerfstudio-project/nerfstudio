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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Optional, Tuple

import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfactory.cameras.rays import RaySamples
from nerfactory.datamanagers.structs import SceneBounds
from nerfactory.fields.instant_ngp_field import TCNNInstantNGPField
from nerfactory.fields.modules.embedding import Embedding
from nerfactory.fields.modules.encoding import Encoding, HashEncoding, SHEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.modules.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfactory.fields.nerf_field import NeRFField
from nerfactory.utils.activations import trunc_exp

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]"""
    return (directions + 1.0) / 2.0


class TCNNCompoundField(TCNNInstantNGPField):
    """Compound Field that uses TCNN"""

    def __init__(
        self,
        aabb,
        num_images: int,
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        appearance_embedding_dim: int = 40,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__(self, aabb)
        self.geo_feat_dim = geo_feat_dim

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)

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
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
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
        # positions = self.spatial_distortion(ray_samples.frustums.get_positions())
        positions = SceneBounds.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        # TODO: add valid_mask masking!
        # tcnn requires directions in the range [0,1]
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        camera_indices = ray_samples.camera_indices.squeeze().to(ray_samples.frustums.origins.device)
        embedded_appearance = self.embedding_appearance(camera_indices)

        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        if density_embedding is None:
            positions = SceneBounds.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            h = torch.cat([d, positions.view(-1, 3), embedded_appearance], dim=-1)
        else:
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim), embedded_appearance], dim=-1)
        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)

        return {FieldHeadNames.RGB: rgb}


class TorchCompoundField(NeRFField):
    """
    PyTorch implementation of the compound field.
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
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__(
            position_encoding,
            direction_encoding,
            base_mlp_num_layers,
            base_mlp_layer_width,
            head_mlp_num_layers,
            head_mlp_layer_width,
            skip_connections,
            spatial_distortion=spatial_distortion,
        )
        self.aabb = Parameter(aabb, requires_grad=False)


field_implementation_to_class = {"tcnn": TCNNCompoundField, "torch": TorchCompoundField}
