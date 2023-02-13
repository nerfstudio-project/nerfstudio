from typing import Callable, Optional

import tinycudann as tcnn
import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.kplanes_field import init_grid_param, interpolate_ms_features


class KPlanesDensityField(Field):
    """TensoRF Field"""

    def __init__(
        self,
        aabb,
        resolution,
        num_input_coords,
        num_output_coords,
        density_activation: Callable,
        spatial_distortion: Optional[SpatialDistortion] = None,
        linear_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.linear_decoder = linear_decoder
        self.hexplane = num_input_coords == 4
        self.feature_dim = num_output_coords
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        activation = "ReLU"
        if self.linear_decoder:
            activation = "None"

        self.grids = init_grid_param(
            grid_nd=2, in_dim=num_input_coords, out_dim=num_output_coords, reso=resolution, a=0.1, b=0.15
        )
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
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
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])

        features = interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None
        )
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            # features.to(pts)
        ).view(n_rays, n_samples, 1)
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> TensorType:
        return torch.zeros(ray_samples.shape)

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[TensorType] = None,
        bg_color: Optional[TensorType] = None,
    ):
        density = self.get_density(ray_samples)

        return {FieldHeadNames.DENSITY: density}

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}
        other_params = {
            k: v for k, v in self.named_parameters() if (k not in nn_params.keys() and k not in field_params.keys())
        }
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
