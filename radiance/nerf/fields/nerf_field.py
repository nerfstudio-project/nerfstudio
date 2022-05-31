""" Classic NeRF field"""


from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from radiance.nerf.field_modules.encoding import Encoding, Identity
from radiance.nerf.field_modules.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from radiance.nerf.field_modules.mlp import MLP
from radiance.nerf.fields.base import Field
from radiance.structures.rays import PointSamples


class NeRFField(Field):
    """NeRF Field"""

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),),
    ) -> None:
        """Create boilerplate NeRF field.

        Args:
            position_encoding (Encoding, optional): Position encoder. Defaults to Identity(in_dim=3).
            direction_encoding (Encoding, optional): Direction encoder. Defaults to Identity(in_dim=3).
            base_mlp_num_layers (int, optional): Number of layers for base MLP. Defaults to 8.
            base_mlp_layer_width (int, optional): Width of base MLP layers. Defaults to 256.
            head_mlp_num_layers (int, optional): Number of layer for ourput head MLP. Defaults to 2.
            head_mlp_layer_width (int, optional): Width of output head MLP layers. Defaults to 128.
            skip_connections (Tuple, optional): Where to add skip connection in base MLP. Defaults to (4,).
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())

    def get_density(self, point_samples: PointSamples):
        """Computes and returns the densities."""
        encoded_xyz = self.position_encoding(point_samples.positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, point_samples: PointSamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        """Computes and returns the outputs."""
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(point_samples.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs
