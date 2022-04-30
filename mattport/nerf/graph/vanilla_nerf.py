"""
Implementation of vanilla nerf.
"""


from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType

from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_heads import DensityFieldHead, FieldHeadNames, RGBFieldHead
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.renderers import RGBRenderer
from mattport.nerf.sampler import PDFSampler, UniformSampler  # pylint: disable=unused-import
from mattport.structures.rays import RaySamples


class NeRFField(nn.Module):
    """NeRF module"""

    def __init__(self, num_layers=8, layer_width=256, skip_connections: Tuple = (4,)) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections

        self.encoding_xyz = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        self.encoding_dir = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )
        self.mlp_base = MLP(
            in_dim=self.encoding_xyz.get_out_dim(),
            out_dim=self.layer_width,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            skip_connections=self.skip_connections,
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


class NeRFGraph(Graph):
    """Vanilla NeRF graph"""

    def __init__(
        self,
        intrinsics=None,
        camera_to_world=None,
        near_plane=1.0,
        far_plane=6.0,
        num_coarse_samples=64,
        num_importance_samples=128,
        **kwargs
    ) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_coarse_samples = num_coarse_samples
        self.num_importance_samples = num_importance_samples
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_modules(self):
        # ray generator
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)

        # samplers
        self.sampler_uniform = UniformSampler(
            near_plane=self.near_plane, far_plane=self.far_plane, num_samples=self.num_coarse_samples
        )
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples)

        # field
        self.field_coarse = NeRFField()
        self.field_fine = NeRFField()

        # renderers
        self.renderer_rgb = RGBRenderer()

        # losses
        self.rgb_loss = MSELoss()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Dict[str, List[Parameter]]: Mapping of different parameter groups
        """
        param_groups = {}
        # param_groups["cameras"] = list(self.ray_generator.parameters())
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def forward(self, ray_indices: TensorType["num_rays", 3]):
        """Takes in the ray indices and renders out values."""
        # get the rays:
        ray_bundle = self.ray_generator.forward(ray_indices)  # RayBundle
        # coarse network:
        uniform_ray_samples = self.sampler_uniform(ray_bundle)  # RaySamples

        # time coarse network # xyz -> asd
        coarse_field_outputs = self.field_coarse(uniform_ray_samples)  # FieldOutputs
        # time end coarse network

        coarse_weights = uniform_ray_samples.get_weights(coarse_field_outputs[FieldHeadNames.DENSITY])

        coarse_renderer_outputs = self.renderer_rgb(
            rgb=coarse_field_outputs[FieldHeadNames.RGB],
            weights=coarse_weights,
        )  # RendererOutputs
        # fine network:
        pdf_ray_samples = self.sampler_pdf(uniform_ray_samples, coarse_weights)  # RaySamples
        fine_field_outputs = self.field_fine(pdf_ray_samples)  # FieldOutputs

        fine_weights = pdf_ray_samples.get_weights(fine_field_outputs[FieldHeadNames.DENSITY])

        fine_renderer_outputs = self.renderer_rgb(
            rgb=fine_field_outputs[FieldHeadNames.RGB],
            weights=fine_weights,
        )  # RendererOutputs
        # outputs:
        outputs = {"rgb_coarse": coarse_renderer_outputs.rgb, "rgb_fine": fine_renderer_outputs.rgb}
        return outputs

    def get_losses(self, batch, graph_outputs):
        # batch.pixels # (num_rays, 3)
        losses = {}
        rgb_loss_coarse = self.rgb_loss(batch["pixels"], graph_outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(batch["pixels"], graph_outputs["rgb_fine"])
        losses = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        return losses
