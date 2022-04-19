"""
Implementation of vanilla nerf.
"""


import torch
from omegaconf import DictConfig
from torch import TensorType, nn

from mattport.nerf.field_modules.encoding import NeRFEncoding
from mattport.nerf.field_modules.field_outputs import DensityFieldOutput, RGBFieldOutput
from mattport.nerf.field_modules.mlp import MLP
from mattport.nerf.field.nerf import NeRFField
from mattport.nerf.field_modules.ray_generator import RayGenerator
from mattport.nerf.graph.base import Graph
from mattport.nerf.loss import MSELoss
from mattport.nerf.renderers import RGBRenderer
from mattport.nerf.sampler import UniformSampler
from mattport.structures.cameras import Rays
from mattport.nerf.dataset.utils import DatasetInputs


class NeRFGraph(Graph):
    def __init__(self, modules_config: DictConfig, dataset_inputs: DatasetInputs) -> None:
        super().__init__(modules_config)

        self.populate_modules(dataset_inputs=dataset_inputs)
        # TODO(ethan): build all modules here instead of in the code below

    def populate_modules(self, dataset_inputs: DatasetInputs):

        # ray generator
        self.ray_generator = RayGenerator(dataset_inputs.intrinsics, dataset_inputs.camera_to_world)

        # samplers
        self.sampler_uniform = UniformSampler(near_plane=0.1, far_plane=4.0, num_samples=64)
        self.sampler_pdf = UniformSampler(near_plane=0.1, far_plane=4.0, num_samples=64)

        # field
        self.coarse = NeRFField()
        self.fine = NeRFField()

        # renderers
        self.renderer_rgb = RGBRenderer()

        # losses
        self.rgb_loss = MSELoss()

    def forward(self, batch_indices: TensorType["num_rays", 3]):

        ray_bundle = self.ray_generator.forward(batch_indices)
        rays = Rays(origin=ray_bundle.origins, direction=ray_bundle.directions)  # TODO(ethan): consolidate this
        ray_samples = self.sampler_uniform(rays)

        xyz = ray_samples.locations
        deltas = ray_samples.deltas

        encoded_xyz = self.encoding_xyz(xyz)
        encoded_dir = self.encoding_dir(rays.direction)
        base_mlp_out = self.mlp_base(encoded_xyz)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))
        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)
        rgb_coarse = self.renderer_rgb(rgb=field_rgb_output, density=field_density_out, deltas=deltas)

        xyz_pdf = self.sampler_pdf(rays)  # TODO(ethan): implement pdf
        encoded_xyz_pdf = self.encoding_xyz(xyz_pdf)
        base_mlp_out = self.mlp_base(encoded_xyz_pdf)
        rgb_mlp_out = self.mlp_rgb(torch.cat([encoded_dir, base_mlp_out], dim=-1))
        field_rgb_output = self.field_output_rgb(rgb_mlp_out)
        field_density_out = self.field_output_density(base_mlp_out)
        rgb_fine = self.renderer_rgb(rgb=field_rgb_output, density=field_density_out, deltas=deltas)

        return {"rgb_coarse": rgb_coarse, "rgb_fine": rgb_fine}

    def get_loss(self, batch, render_outputs):
        rgb_loss_coarse = self.rgb_loss(batch.rgb, render_outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(batch.rgb, render_outputs["rgb_fine"])
        return {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
