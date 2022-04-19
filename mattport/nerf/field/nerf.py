"""
Multi Layer Perceptron
"""
from typing import Optional

from torch import nn
from torchtyping import TensorType
from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_outputs import DensityFieldOutput, RGBFieldOutput
from mattport.nerf.field_modules.mlp import MLP


class NeRFField():
    """Multilayer perceptron"""

    def __init__(self) -> None:
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize mulilayer perceptron."""
        self.encoding_xyz = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0)
        self.encoding_dir = NeRFEncoding(in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=4.0)
        self.mlp_base = MLP(
            in_dim=self.encoding_xyz.get_out_dim(), out_dim=64, num_layers=8, layer_width=64, activation=nn.ReLU()
        )
        self.mlp_rgb = MLP(
            in_dim=self.base_mlp.get_out_dim() + self.encoding_dir.get_out_dim(),
            out_dim=64,
            num_layers=2,
            layer_width=64,
            activation=nn.ReLU(),
        )
        self.field_output_rgb = RGBFieldOutput(in_dim=self.mlp_rgb.get_out_dim())
        self.field_output_density = DensityFieldOutput(in_dim=self.mlp_base.get_out_dim())

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> TensorType[..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Network output
        """

        encoded_xyz = self.encoding_xyz(xyz)
        encoded_dir = self.encoding_dir(rays.direction)
        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))
        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)
        return rgb_coarse
