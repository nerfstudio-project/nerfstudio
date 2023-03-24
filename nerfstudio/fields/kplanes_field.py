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
Fields for K-Planes (https://sarafridov.github.io/K-Planes/).
"""

from dataclasses import field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import KPlanesEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

CONSOLE = Console(width=120)


def interpolate_ms_features(
    pts: torch.Tensor,
    grid_encodings: Iterable[KPlanesEncoding],
    concat_features: bool,
) -> torch.Tensor:
    """Combines/interpolates features across multiple dimensions and scales.
    Args:
        pts: Coordinates to query
        grid_encodings: Grid encodings to query
        concat_features: Whether to concatenate features at different scales
    Returns:
        Feature vectors
    """

    multi_scale_interp = [] if concat_features else 0.0
    grid: nn.ParameterList
    for grid in grid_encodings:
        grid_features = grid(pts)

        # combine over scales
        if concat_features:
            multi_scale_interp.append(grid_features)
        else:
            multi_scale_interp = multi_scale_interp + grid_features

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp


class KPlanesField(Field):
    """K-Planes field.

    Args:
        aabb: Parameters of scene aabb bounds
        num_images: How many images exist in the dataset
        geo_feat_dim: Output geo feat dimensions
        grid_base_resolution: Base grid resolution
        grid_feature_dim: Dimension of feature vectors stored in grid
        concat_features_across_scales: Whether to concatenate features at different scales
        multiscale_res: Multiscale grid resolutions
        spatial_distortion: Spatial distortion to apply to the scene
        appearance_embedding_dim: Dimension of appearance embedding. Set to 0 to disable
        use_average_appearance_embedding: Whether to use average appearance embedding or zeros for inference
        linear_decoder: Whether to use a linear decoder instead of an MLP
        linear_decoder_layers: Number of layers in linear decoder
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        geo_feat_dim: int = 15,
        concat_features_across_scales: bool = True,
        grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128]),
        grid_feature_dim: int = 32,
        multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4]),
        spatial_distortion: Optional[SpatialDistortion] = None,
        appearance_embedding_dim: int = 0,
        use_average_appearance_embedding: bool = True,
        linear_decoder: bool = False,
        linear_decoder_layers: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.register_buffer("aabb", aabb)
        self.num_images = num_images
        self.geo_feat_dim = geo_feat_dim
        self.grid_base_resolution = grid_base_resolution
        self.concat_features_across_scales = concat_features_across_scales
        self.spatial_distortion = spatial_distortion

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in multiscale_res:
            # initialize coordinate grid
            # Resolution fix: multi-res only on spatial planes
            resolution = [r * res for r in grid_base_resolution[:3]] + grid_base_resolution[3:]
            gp = KPlanesEncoding(resolution, grid_feature_dim)

            # Concatenate over feature len for each scale
            if self.concat_features_across_scales:
                self.feature_dim += grid_feature_dim
            else:
                self.feature_dim = grid_feature_dim

            self.grids.append(gp)

        # 2. Init appearance code-related parameters
        self.appearance_embedding_dim = appearance_embedding_dim
        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_average_appearance_embedding = use_average_appearance_embedding  # for test-time

        # 3. Init decoder params
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.linear_decoder = linear_decoder
        # 4. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,  # self.direction_encoder.n_output_dims,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = (
                self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        grid_input = ray_samples.frustums.get_positions().view(-1, 3)

        if self.spatial_distortion is not None:
            grid_input = self.spatial_distortion(grid_input)
            grid_input = grid_input / 2  # from [-2, 2] to [-1, 1]
        else:
            # Input should be in [-1, 1]
            grid_input = (grid_input - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1.0

        if len(self.grid_base_resolution) == 4:
            grid_input = torch.cat([grid_input, (2 * ray_samples.times - 1).reshape(-1, 1)], -1)

        features = interpolate_ms_features(
            grid_input, grid_encodings=self.grids, concat_features=self.concat_features_across_scales
        )

        if self.linear_decoder:
            density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        else:
            features = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(grid_input) - 1)
        return density, features

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        directions = ray_samples.frustums.directions.reshape(-1, 3)

        if self.linear_decoder:
            color_features = [density_embedding]
        else:
            directions = shift_directions_for_tcnn(directions)
            d = self.direction_encoding(directions)
            color_features = [d, density_embedding.view(-1, self.geo_feat_dim)]

        if self.appearance_embedding_dim > 0:
            if self.training:
                camera_indices = ray_samples.camera_indices.squeeze()
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (ray_samples.frustums.directions.shape[0], self.appearance_embedding_dim),
                        device=directions.device,
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (ray_samples.frustums.directions.shape[0], self.appearance_embedding_dim),
                        device=directions.device,
                    )

            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if self.linear_decoder:
            if self.appearance_embedding_dim > 0:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]

            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = torch.sigmoid(rgb).view(*outputs_shape, -1).to(directions)
        else:
            rgb = self.color_net(color_features).view(*outputs_shape, -1)

        return {FieldHeadNames.RGB: rgb}


class KPlanesDensityField(Field):
    """A lightweight density field module.
    Args:
        aabb: Parameters of scene aabb bounds
        resolution: Grid resolution
        num_output_coords: dimension of grid feature vectors
        spatial_distortion: Spatial distortion to apply to the scene
        linear_decoder: Whether to use a linear decoder instead of an MLP
    """

    def __init__(
        self,
        aabb: TensorType,
        resolution: List[int],
        num_output_coords: int,
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)

        self.spatial_distortion = spatial_distortion
        self.hexplane = len(resolution) == 4
        self.feature_dim = num_output_coords
        self.linear_decoder = linear_decoder

        self.grids = KPlanesEncoding(resolution, num_output_coords, init_a=0.1, init_b=0.15)

        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if self.linear_decoder else "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        CONSOLE.log(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - resolution={resolution}")

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        """Computes and returns the densities."""
        grid_input = ray_samples.frustums.get_positions().view(-1, 3)

        if self.spatial_distortion is not None:
            grid_input = self.spatial_distortion(grid_input)
            grid_input = grid_input / 2
        else:
            grid_input = (grid_input - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1.0

        if self.hexplane:
            grid_input = torch.cat([grid_input, (2 * ray_samples.times - 1).view(-1, 1)], -1)

        features = interpolate_ms_features(grid_input, grid_encodings=[self.grids], concat_features=False)

        density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        density = trunc_exp(density_before_activation.to(grid_input) - 1)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}
