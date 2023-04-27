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
DreamNGP field implementations using tiny-cuda-nn, torch, ....
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from nerfacc import ContractionType, contract
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
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


class DreamNGPField(TCNNInstantNGPField):
    """TCNN implementation of the Instant-NGP field.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        use_appearance_embedding: whether to use appearance embedding
        num_images: number of images, required if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
        contraction_type: type of contraction
        num_levels: number of levels of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
    """

    def __init__(
        self,
        aabb: TensorType,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        use_appearance_embedding: Optional[bool] = False,
        num_images: Optional[int] = None,
        appearance_embedding_dim: int = 32,
        contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE,
        num_levels: int = 16,
        log2_hashmap_size: int = 19,
        max_res: int = 2048,
    ) -> None:
        super().__init__(
            aabb,
            num_layers,
            hidden_dim,
            geo_feat_dim,
            num_layers_color,
            hidden_dim_color,
            use_appearance_embedding,
            num_images,
            appearance_embedding_dim, 
            contraction_type,
            num_levels,
            log2_hashmap_size,
            max_res
        )
        self.mlp_background_color = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 32,
                "n_hidden_layers": 1,
            },
        )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities"""
        positions = ray_samples.frustums.get_positions()
        # positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)

        self._sample_locations = positions_flat
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # import pdb 
        # pdb.set_trace()

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out



    def get_background_rgb(self, ray_bundle: RayBundle):
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)

        outputs_shape = ray_bundle.directions.shape[:-1]
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        background_rgb = self.mlp_background_color(directions_flat).view(*outputs_shape, -1).to(directions)

        return background_rgb