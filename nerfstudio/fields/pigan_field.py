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

"""Pi-GAN Generator = field"""


from typing import Dict, Optional, Tuple, TypedDict

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP, SIREN
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.latent import FilmLatent

# NOTE -  Just Add latent



class PiganField(Field):
    """PiganField

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 1,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        siren_omega: int = 30,
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.siren_omega = siren_omega
        self.spatial_distortion = spatial_distortion
        self.base_mlp_num_layers = base_mlp_num_layers


        self.mlp_base = SIREN(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            film=True,
            first_omega=siren_omega,
            hidden_omega=siren_omega 
        )

        self.mlp_head = SIREN(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            film=True,
            first_omega=siren_omega,
            hidden_omega=siren_omega 
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    # NOTE - 여기 들어오는 latent들은 모두 mapping network를 거쳐서 여러 layer의 latent 값들이 다 담긴거.
    def get_density(self, ray_samples: RaySamples, latents: FilmLatent):
        if self.use_integrated_encoding:
            gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            if self.spatial_distortion is not None:
                gaussian_samples = self.spatial_distortion(gaussian_samples)
            encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions = ray_samples.frustums.get_positions()
            if self.spatial_distortion is not None:
                positions = self.spatial_distortion(positions)
            encoded_xyz = self.position_encoding(positions)

        base_mlp_out = self.mlp_base(encoded_xyz, latents)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, latents: FilmLatent, density_embedding: Optional[TensorType] = None, 
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1),latents)  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs

    def forward(self, ray_samples: RaySamples, latents: FilmLatent, compute_normals: bool = False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples, latents)
        else:
            density, density_embedding = self.get_density(ray_samples, latents)

        latent = {k:v[-1] for k,v in latents.items()}
        field_outputs = self.get_outputs(ray_samples, latent, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs