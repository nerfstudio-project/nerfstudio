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

import itertools
from dataclasses import field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

CONSOLE = Console(width=120)


def init_grid_param(out_dim: int, reso: List[int], a: float = 0.1, b: float = 0.5):
    """Initializes the grid parameters."""

    in_dim = len(reso)
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), 2))
    grid_coefs = nn.ParameterList()
    for coo_comb in coo_combs:
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


@torch.jit.script
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """Generates feature vectors by sampling and interpolating between grid values."""

    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    assert grid_dim in (
        2,
        3,
    ), f"Grid-sample was called with {grid_dim}D data but is only implemented for 2 and 3D data."

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]

    interp = F.grid_sample(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )

    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]

    return interp


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Union[nn.ModuleList, List[nn.ParameterList]],
    concat_features: bool,
) -> torch.Tensor:
    """Combines/interpolates features across multiple dimensions and scales."""

    coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))

    multi_scale_interp = [] if concat_features else 0.0
    grid: nn.ParameterList
    for grid in ms_grids:
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp


class KPlanesField(Field):
    """K-Planes field."""

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        geo_feat_dim: int = 15,
        grid_config: List[Dict] = field(default_factory=lambda: []),
        concat_features_across_scales: bool = True,
        multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4, 8]),
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
        self.grid_config = grid_config
        self.concat_features_across_scales = concat_features_across_scales
        self.spatial_distortion = spatial_distortion

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in multiscale_res:
            # initialize coordinate grid
            config = grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gp = init_grid_param(out_dim=config["output_coordinate_dim"], reso=config["resolution"])

            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features_across_scales:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]

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

        if len(self.grid_config[0]["resolution"]) == 4:
            grid_input = torch.cat([grid_input, (2 * ray_samples.times - 1).reshape(-1, 1)], -1)

        features = interpolate_ms_features(
            grid_input, ms_grids=self.grids, concat_features=self.concat_features_across_scales
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
    """A lightweight density field module."""

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

        self.grids = init_grid_param(out_dim=num_output_coords, reso=resolution, a=0.1, b=0.15)

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

        if ray_samples.times is not None and self.hexplane:
            grid_input = torch.cat([grid_input, (2 * ray_samples.times - 1).view(-1, 1)], -1)

        features = interpolate_ms_features(grid_input, ms_grids=[self.grids], concat_features=False)

        density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        density = trunc_exp(density_before_activation.to(grid_input) - 1)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}
