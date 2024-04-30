# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field


class VanillaNerfWField(Field):
    """The NeRF-W field which has appearance and transient conditioning.

    Args:
        num_images: How many images exist in the dataset.
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        appearance_embedding_dim:: Dimension of appearance embedding.
        transient_embedding_dim:: Dimension of transient embedding.
        skip_connections: Where to add skip connection in base MLP.
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

        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.mlp_transient = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.embedding_transient.get_out_dim(),
            out_dim=base_mlp_layer_width // 2,
            num_layers=4,
            layer_width=base_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim()
            + self.direction_encoding.get_out_dim()
            + (self.embedding_appearance.get_out_dim() if self.embedding_appearance is not None else 0),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_head_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_head_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())

        self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
        self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.get_out_dim())
        self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.get_out_dim())

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_head_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Float[Tensor, "*batch embedding_size"]] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """Returns the outputs of the NeRF-W field.

        Args:
            ray_samples: Ray samples.
            density_embedding: Density embedding.

        Returns:
            Outputs of the NeRF-W field.
        """
        outputs = {}
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze().to(ray_samples.frustums.origins.device)
        mlp_in = [density_embedding, encoded_dir]
        if self.embedding_appearance is not None:
            embedded_appearance = self.embedding_appearance(camera_indices)
            mlp_in.append(embedded_appearance)
        mlp_head_out = self.mlp_head(torch.cat(mlp_in, dim=-1))
        outputs[self.field_head_rgb.field_head_name] = self.field_head_rgb(mlp_head_out)  # static rgb
        embedded_transient = self.embedding_transient(camera_indices)
        transient_mlp_in = torch.cat([density_embedding, embedded_transient], dim=-1)  # type: ignore
        transient_mlp_out = self.mlp_transient(transient_mlp_in)
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
