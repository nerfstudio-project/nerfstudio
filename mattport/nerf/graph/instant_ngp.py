"""
Implementation of Instant NGP.
"""

import torch
from torch import nn

from mattport.nerf.field_modules.encoding import HashEncoding, NeRFEncoding
from mattport.nerf.field_modules.field_heads import DensityFieldHead, RGBFieldHead
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.graph.vanilla_nerf import NeRFGraph
from mattport.structures.rays import RaySamples


class NGPField(nn.Module):
    """NeRF module"""

    def __init__(self, num_layers=3, layer_width=64) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.encoding_xyz = HashEncoding()
        self.encoding_dir = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )
        self.mlp_base = MLP(
            in_dim=self.encoding_xyz.get_out_dim(),
            out_dim=self.layer_width,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            activation=nn.ReLU(),
        )
        self.mlp_rgb = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.encoding_dir.get_out_dim(),
            out_dim=self.layer_width // 2,
            num_layers=1,
            activation=nn.ReLU(),
        )
        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_rgb.get_out_dim())
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray
        Args:
            xyz: ()
        # TODO(ethan): change the input to be something more abstracted
        e.g., a FieldInput structure
        """
        positions = ray_samples.positions
        directions = ray_samples.directions
        encoded_xyz = self.encoding_xyz(positions)
        encoded_dir = self.encoding_dir(directions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))

        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)

        field_outputs = {}
        field_outputs.update(field_rgb_output)
        field_outputs.update(field_density_out)
        return field_outputs


class NGPGraph(NeRFGraph):
    """NeRF-W graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, **kwargs) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """Set the fields."""
        self.field_coarse = NGPField()
        self.field_fine = NGPField()
