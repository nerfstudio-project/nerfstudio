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
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
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


class DreamNGPField(TCNNNerfactoField):
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
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__(
            aabb, 
            num_images, 
            log2_hashmap_size=log2_hashmap_size,
            max_res=max_res,
            spatial_distortion=None
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

    def get_background_rgb(self, ray_bundle: RayBundle):
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)

        outputs_shape = ray_bundle.directions.shape[:-1]
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        background_rgb = self.mlp_background_color(directions_flat).view(*outputs_shape, -1).to(directions)

        return background_rgb