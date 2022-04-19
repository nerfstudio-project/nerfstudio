"""
Multi Layer Perceptron
"""
from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType

from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_outputs import DensityFieldOutput, RGBFieldOutput
from mattport.nerf.field_modules.mlp import MLP
from mattport.structures.rays import RayBundle, RaySamples


class NeRFField(nn.Module):
    """Multilayer perceptron"""

    def __init__(self) -> None:
        super().__init__()
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize modules."""
        self.encoding_xyz = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0)
        self.encoding_dir = NeRFEncoding(in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=4.0)
        self.mlp_base = MLP(
            in_dim=self.encoding_xyz.get_out_dim(), out_dim=64, num_layers=8, layer_width=64, activation=nn.ReLU()
        )
        self.mlp_rgb = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.encoding_dir.get_out_dim(),
            out_dim=64,
            num_layers=2,
            layer_width=64,
            activation=nn.ReLU(),
        )
        self.field_output_rgb = RGBFieldOutput(in_dim=self.mlp_rgb.get_out_dim())
        self.field_output_density = DensityFieldOutput(in_dim=self.mlp_base.get_out_dim())

    def forward(self, xyz: TensorType["num_rays", 3], dirs: TensorType["num_rays", 3]):
        """

        Args:
            xyz: ()
        # TODO(ethan): change the input to be something more abstracted
        e.g., a FieldInput structure
        """

        encoded_xyz = self.encoding_xyz(xyz)
        encoded_dir = self.encoding_dir(dirs)
        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))
        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)
        return field_rgb_output, field_density_out
