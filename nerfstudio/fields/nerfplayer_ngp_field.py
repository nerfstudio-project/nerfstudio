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
NeRFPlayer (https://arxiv.org/abs/2210.15947) field implementations with InstantNGP backbone
"""


from typing import Dict, Optional, Tuple

import torch
from nerfacc import ContractionType, contract
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_grid import TemporalGridEncoder
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class NerfplayerNGPField(Field):
    """NeRFPlayer (https://arxiv.org/abs/2210.15947) field with InstantNGP backbone.

    Args:
        aabb: parameters of scene aabb bounds
        temporal_dim: the dimension of temporal axis, a higher dimension indicates a higher temporal frequency
            please refer to the implementation of TemporalGridEncoder for more details
        num_levels: the number of multi-resolution levels; same as InstantNGP
        features_per_level: the dim of output feature vector for each level; same as InstantNGP
        log2_hashmap_size: the size of the table; same as InstantNGP
        base_resolution: base resolution for the table; same as InstantNGP
        num_layers: number of hidden layers (occupancy decoder network after sampling)
        hidden_dim: dimension of hidden layers (occupancy decoder network after sampling)
        geo_feat_dim: output geo feat dimensions
        num_layers_color: number of hidden layers for color network
        hidden_dim_color: dimension of hidden layers for color network
        use_appearance_embedding: whether to use appearance embedding
        disable_viewing_dependent: if true, disable the viewing dependent effect (no viewing direction as inputs)
            Sometimes we need to disable viewing dependent effects in a dynamic scene, because there is
            ambiguity between being dynamic and viewing dependent effects. For example, the shadow of the camera
            should be a dynamic effect, but may be reconstructed as viewing dependent effects.
        num_images: number of images, requried if use_appearance_embedding is True
        appearance_embedding_dim: dimension of appearance embedding
        contraction_type: type of contraction
    """

    def __init__(
        self,
        aabb: TensorType,
        temporal_dim: int = 16,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        use_appearance_embedding: bool = False,
        disable_viewing_dependent: bool = False,
        num_images: Optional[int] = None,
        appearance_embedding_dim: int = 32,
        contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE,
    ) -> None:
        super().__init__()

        self.aabb = Parameter(aabb, requires_grad=False)
        self.geo_feat_dim = geo_feat_dim
        self.contraction_type = contraction_type

        self.use_appearance_embedding = use_appearance_embedding
        if use_appearance_embedding:
            assert num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = Embedding(num_images, appearance_embedding_dim)

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_base = TemporalGridEncoder(
            input_dim=3,
            temporal_dim=temporal_dim,
            num_levels=num_levels,
            level_dim=features_per_level,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=1024 * (self.aabb.max() - self.aabb.min()),
        )
        self.mlp_base_decode = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        in_dim = self.direction_encoding.n_output_dims + self.geo_feat_dim
        if disable_viewing_dependent:
            in_dim = self.geo_feat_dim
            self.direction_encoding = None
        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
        self.mlp_head = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        positions = ray_samples.frustums.get_positions()
        positions_flat = positions.view(-1, 3)
        positions_flat = contract(x=positions_flat, roi=self.aabb, type=self.contraction_type)
        assert ray_samples.times is not None, "Time should be included in the input for NeRFPlayer"
        times_flat = ray_samples.times.view(-1, 1)

        h = self.mlp_base(positions_flat, times_flat)
        h = self.mlp_base_decode(h).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        if self.direction_encoding is not None:
            d = self.direction_encoding(directions_flat)
            if density_embedding is None:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
                h = torch.cat([d, positions.view(-1, 3)], dim=-1)
            else:
                h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        else:
            # viewing direction is disabled
            if density_embedding is None:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
                h = positions.view(-1, 3)
            else:
                h = density_embedding.view(-1, self.geo_feat_dim)

        if self.use_appearance_embedding:
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )
            h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*ray_samples.frustums.directions.shape[:-1], -1).to(directions)
        return {FieldHeadNames.RGB: rgb}

    # pylint: disable=arguments-differ
    def density_fn(self, positions: TensorType["bs":..., 3], times: TensorType["bs":..., 1]) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.
        Overwrite this function since density is time dependent now.

        Args:
            positions: the origin of the samples/frustums
            times: the time of each position
        """
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_opacity(self, positions: TensorType["bs":..., 3], step_size, time_intervals=10) -> TensorType["bs":..., 1]:
        """Returns the opacity for a position and time. Used primarily by the occupancy grid.
        This will return the maximum opacity of the points in the space in a dynamic sequence.

        Args:
            positions: the positions to evaluate the opacity at.
            step_size: the step size to use for the opacity evaluation.
            time_intervals: sample density on N time stamps
        """
        # TODO: Converting opacity by time intervals is slow, and may lead to temporal artifacts.
        #       (Maybe random sample time and EMA?)
        opacity = []
        for t in range(0, time_intervals):
            density = self.density_fn(positions, t / (time_intervals - 1) * torch.ones_like(positions)[..., [0]])
            opacity.append(density * step_size)
        opacity = torch.stack(opacity, dim=0).max(dim=0).values
        return opacity
