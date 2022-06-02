"""Fields for nerf-w"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from pyrad.nerf.field_modules.embedding import Embedding
from pyrad.nerf.field_modules.encoding import Encoding, Identity
from pyrad.nerf.field_modules.field_heads import (
    DensityFieldHead,
    FieldHeadNames,
    RGBFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from pyrad.nerf.field_modules.mlp import MLP
from pyrad.nerf.fields.base import Field
from pyrad.structures.rays import PointSamples


class VanillaNerfWField(Field):
    """The NeRF-W field which has appearance and transient conditioning."""

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

    def get_density(self, point_samples: PointSamples):
        """Computes and returns the densities."""
        encoded_xyz = self.position_encoding(point_samples.positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_head_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, point_samples: PointSamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        encoded_dir = self.direction_encoding(point_samples.directions)
        embedded_appearance = self.embedding_appearance(point_samples.camera_indices)
        mlp_head_out = self.mlp_head(torch.cat([density_embedding, encoded_dir, embedded_appearance], dim=-1))
        outputs[self.field_head_rgb.field_head_name] = self.field_head_rgb(mlp_head_out)  # static rgb
        embedded_transient = self.embedding_transient(point_samples.camera_indices)
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
