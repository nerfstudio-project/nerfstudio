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

"""Fields for nerf-w"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from pyrad.fields.modules.embedding import Embedding
from pyrad.fields.modules.encoding import Encoding, Identity
from pyrad.fields.modules.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from pyrad.fields.modules.mlp import MLP
from pyrad.fields.base import Field
from pyrad.cameras.rays import RaySamples


class VanillaNerfWField(Field):
    """The NeRF-W field which has appearance and transient conditioning.

    Args:
        num_images (int): How many images exist in the dataset.
        position_encoding (Encoding, optional): Position encoder. Defaults to Identity(in_dim=3).
        direction_encoding (Encoding, optional): Direction encoder. Defaults to Identity(in_dim=3).
        base_mlp_num_layers (int, optional): Number of layers for base MLP. Defaults to 8.
        base_mlp_layer_width (int, optional): Width of base MLP layers. Defaults to 256.
        head_mlp_num_layers (int, optional): Number of layer for ourput head MLP. Defaults to 2.
        head_mlp_layer_width (int, optional): Width of output head MLP layers. Defaults to 128.
        appearance_embedding_dim: (int, optional): Dimension of appearance embedding. Defaults to 48.
        transient_embedding_dim: (int, optional): Dimension of transient embedding. Defaults to 16.
        skip_connections (Tuple, optional): Where to add skip connection in base MLP. Defaults to (4,).
    """

    def __init__(
        self,
        num_images: int,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        appearance_embedding_dim: int = 48,
        transient_embedding_dim: int = 16,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.num_images = num_images
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.base_mlp_num_layers = base_mlp_num_layers
        self.base_mlp_layer_width = base_mlp_layer_width
        self.head_mlp_num_layers = head_mlp_num_layers
        self.head_mlp_layer_width = head_mlp_layer_width
        self.appearance_embedding_dim = appearance_embedding_dim
        self.transient_embedding_dim = transient_embedding_dim

        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
        )
        self.mlp_transient = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.embedding_transient.get_out_dim(),
            out_dim=base_mlp_layer_width // 2,
            num_layers=4,
            layer_width=base_mlp_layer_width,
            activation=nn.ReLU(),
        )
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim()
            + self.direction_encoding.get_out_dim()
            + self.embedding_appearance.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
        )

        self.field_head_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_head_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())

        self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
        self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.get_out_dim())
        self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.get_out_dim())

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_head_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType[..., "embedding_size"]] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        """Returns the outputs of the NeRF-W field.

        Args:
            ray_samples (RaySamples): Ray samples.
            density_embedding (TensorType[..., "embedding_size"], optional): Density embedding. Defaults to None.

        Returns:
            Dict[FieldHeadNames, TensorType]: Outputs of the NeRF-W field.
        """
        outputs = {}
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        embedded_appearance = self.embedding_appearance(ray_samples.camera_indices.squeeze())
        mlp_head_out = self.mlp_head(torch.cat([density_embedding, encoded_dir, embedded_appearance], dim=-1))
        outputs[self.field_head_rgb.field_head_name] = self.field_head_rgb(mlp_head_out)  # static rgb
        embedded_transient = self.embedding_transient(ray_samples.camera_indices.squeeze())
        transient_mlp_out = self.mlp_transient(torch.cat([density_embedding, embedded_transient], dim=-1))
        outputs[self.field_head_transient_uncertainty.field_head_name] = self.field_head_transient_uncertainty(
            transient_mlp_out
        )  # uncertainty
        outputs[self.field_head_transient_rgb.field_head_name] = self.field_head_transient_rgb(
            transient_mlp_out
        )  # transient rgb
        outputs[self.field_head_transient_density.field_head_name] = self.field_head_transient_density(
            transient_mlp_out
        )  # transient density
        return outputs
