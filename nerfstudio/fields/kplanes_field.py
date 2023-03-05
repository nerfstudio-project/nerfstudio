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

"""KPlanes Field"""


import itertools
from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Union

import tinycudann as tcnn
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.interpolation import grid_sample_wrapper


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]
    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    """TODO"""
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(grid_nd: int, in_dim: int, out_dim: int, reso: Sequence[int], a: float = 0.1, b: float = 0.5):
    """TODO"""
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    grid_dimensions: int,
    concat_features: bool,
    num_levels: Optional[int],
) -> torch.Tensor:
    """todo"""
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.0
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):  # type: ignore
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)  # type: ignore
        else:
            multi_scale_interp = multi_scale_interp + interp_space  # type: ignore

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)  # type: ignore
    return multi_scale_interp  # type: ignore


class KPlanesField(Field):
    """TensoRF Field"""

    def __init__(
        self,
        aabb,
        # the aabb bounding box of the dataset
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        head_mlp_num_layers: int = 2,
        # number of layers for the MLP
        head_mlp_layer_width: int = 128,
        # layer width for the MLP
        use_sh: bool = False,
        # whether to use spherical harmonics as the feature decoding function
        sh_levels: int = 2,
        # number of levels to use for spherical harmonics
        # added stuff
        spatial_distortion: Optional[SpatialDistortion] = None,
        grid_config: Union[str, List[Dict]] = "",
        num_images: int = 0,
        density_activation: Optional[Callable] = None,
        multiscale_res: Optional[Sequence[int]] = None,
        concat_features_across_scales: bool = False,
        linear_decoder: bool = True,
        linear_decoder_layers: Optional[int] = None,
        use_appearance_embedding: bool = False,
        disable_viewing_dependent: bool = False,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config
        self.num_images = num_images

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]  # type: ignore

        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0

        for res in self.multiscale_res_multipliers:

            config = self.grid_config[0].copy()  # type: ignore
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)

        # appearance embeddings
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding

        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # Lambertian effects
        self.disable_viewing_dependent = disable_viewing_dependent

        if not disable_viewing_dependent:
            # Init direction encoder
            self.direction_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        # Init decoder network
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
            self.geo_feat_dim = 15
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
            self.in_dim_color = self.geo_feat_dim + self.appearance_embedding_dim
            if not disable_viewing_dependent:
                self.in_dim_color += self.direction_encoder.n_output_dims
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

    def get_density(self, ray_samples: RaySamples):

        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        pts = positions
        n_rays, n_samples = pts.shape[:2]

        timestamps = ray_samples.times
        if timestamps is not None:
            # print(pts.shape, timestamps.shape)
            # timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])

        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,  # type: ignore
            grid_dimensions=self.grid_config[0]["grid_dimensions"],  # type: ignore
            concat_features=self.concat_features,
            num_levels=None,
        )

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        density = self.density_activation(density_before_activation.to(pts)).view(n_rays, n_samples, 1)  # type: ignore
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> TensorType:

        pts: torch.Tensor = ray_samples.frustums.get_positions()
        n_rays, n_samples = pts.shape[:2]

        if ray_samples.camera_indices is not None:
            camera_indices = ray_samples.camera_indices.squeeze()
        directions: torch.Tensor = get_normalized_directions(ray_samples.frustums.directions)
        # directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        if not self.linear_decoder and not self.disable_viewing_dependent:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
            color_features = [encoded_directions, density_embedding.view(-1, self.geo_feat_dim)]
        else:
            color_features = [density_embedding]

        color_features = torch.cat(color_features, dim=-1)  # type: ignore

        if self.linear_decoder:
            basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        return rgb  # type: ignore

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[TensorType] = None,
        bg_color: Optional[TensorType] = None,
    ):
        density, density_features = self.get_density(ray_samples)
        rgb = self.get_outputs(ray_samples, density_features)  # type: ignore

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        if self.disable_viewing_dependent:
            nn_params = [self.sigma_net.named_parameters(prefix="sigma_net")]
        else:
            nn_params = [
                self.sigma_net.named_parameters(prefix="sigma_net"),
                self.direction_encoder.named_parameters(prefix="direction_encoder"),
            ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {
            k: v for k, v in self.named_parameters() if (k not in nn_params.keys() and k not in field_params.keys())
        }
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
