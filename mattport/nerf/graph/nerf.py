"""
Implementation of vanilla nerf.
"""


import logging

import torch
from omegaconf import DictConfig
from torch import nn
from torchtyping import TensorType

from mattport.nerf.dataset.utils import DatasetInputs, get_dataset_inputs
from mattport.nerf.field.nerf import NeRFField
from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_outputs import DensityFieldOutput, RGBFieldOutput
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.renderers import RGBRenderer
from mattport.nerf.sampler import UniformSampler


class NeRFGraph(Graph):
    def __init__(self, intrinsics=None, camera_to_world=None) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world)

    def populate_modules(self):
        # ray generator
        self.ray_generator = RayGenerator(self.intrinsics, self.camera_to_world)

        # samplers
        self.sampler_uniform = UniformSampler(near_plane=0.1, far_plane=4.0, num_samples=64)
        self.sampler_pdf = UniformSampler(near_plane=0.1, far_plane=4.0, num_samples=64)

        # field
        self.field_coarse = NeRFField()
        self.field_fine = NeRFField()

        # renderers
        self.renderer_rgb = RGBRenderer()

        # losses
        self.rgb_loss = MSELoss()

    def forward(self, ray_indices: TensorType["num_rays", 3]):
        """Takes in the ray indices and renders out values."""

        ray_bundle = self.ray_generator.forward(ray_indices)
        ray_samples = self.sampler_uniform(ray_bundle)

        xyz = ray_samples.get_positions(ray_bundle) # (num_rays, num_samples, 3)
        num_rays, num_samples = xyz.shape[:2]
        xyz = xyz.view(-1, 3)
        dirs = (
            ray_bundle.directions.unsqueeze(1).repeat(1, num_samples, 1).view(num_rays * num_samples, 3)
        )  # (num_rays, 3)
        
        field_output_rgb, field_output_density = self.field_coarse(xyz=xyz, dirs=dirs)
        # move these back to the correct shape
        field_output_rgb = field_output_rgb.view(num_rays, num_samples, 3)
        field_output_density = field_output_density.view(num_rays, num_samples)

        deltas = ray_samples.get_deltas()
        rgb_coarse = self.renderer_rgb(rgb=field_output_rgb, density=field_output_density, deltas=deltas)
        
        print(field_output_rgb)
        print(field_output_density)
        outputs = {
            "rgb_coarse": rgb_coarse
        }
        return outputs

    def get_losses(self, batch, graph_outputs):
        # batch.pixels # (num_rays, 3)
        losses = {}
        # rgb_loss_coarse = self.rgb_loss(batch.pixels, graph_outputs["rgb_coarse"])
        # rgb_loss_fine = self.rgb_loss(batch.pixels, graph_outputs["rgb_fine"])
        # losses = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        return losses
