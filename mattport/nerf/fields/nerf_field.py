""" Classic NeRF field"""


from typing import Tuple

import torch
from torch import nn

from mattport.nerf.field_modules.encoding import Encoding, Identity
from mattport.nerf.field_modules.field_heads import DensityFieldHead, FieldHead, RGBFieldHead
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.fields.base import Field
from mattport.structures.rays import PointSamples
from mattport.utils.misc import is_not_none


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
        positions = point_samples.positions
        valid_mask = point_samples.valid_mask
        if not is_not_none(valid_mask):
            valid_mask = torch.ones_like(positions[..., 0]).bool()
        # placeholders for values to return
        density = torch.zeros(*valid_mask.shape, 1, dtype=torch.float32, device=positions.device)
        base_mlp_out = torch.zeros(
            *valid_mask.shape, self.mlp_base.out_dim, dtype=torch.float32, device=positions.device
        )
        if not valid_mask.any():  # empty mask
            return density, base_mlp_out

        encoded_xyz = self.position_encoding(positions[valid_mask])
        base_mlp_out[valid_mask] = self.mlp_base(encoded_xyz)
        density[valid_mask] = self.field_output_density(base_mlp_out[valid_mask])
        return density, base_mlp_out

    def get_outputs(self, point_samples: PointSamples, density_embedding=None, valid_mask=None):
        directions = point_samples.directions
        if not is_not_none(valid_mask):
            valid_mask = torch.ones_like(directions[..., 0]).bool()
        outputs = {}
        for field_head in self.field_heads:
            # placeholders for values to return
            out = torch.zeros(*valid_mask.shape, field_head.out_dim, dtype=torch.float32, device=directions.device)
            if not valid_mask.any():  # empty mask
                return {field_head.field_head_name: out}
            encoded_dir = self.direction_encoding(directions[valid_mask])
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding[valid_mask]], dim=-1))
            out[valid_mask] = field_head(mlp_out)
            outputs[field_head.field_head_name] = out
        return outputs
