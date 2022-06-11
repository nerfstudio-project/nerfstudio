# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""
Implementation of Instant NGP.
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pyrad.fields.modules.field_heads import FieldHeadNames
from pyrad.fields.instant_ngp_field import field_implementation_to_class
from pyrad.graphs.base import Graph
from pyrad.optimizers.loss import MSELoss
from pyrad.fields.occupancy_fields.occupancy_grid import OccupancyGrid
from pyrad.renderers.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from pyrad.graphs.modules.ray_sampler import PDFSampler, UniformSampler
from pyrad.utils import colors
from pyrad.cameras.rays import RayBundle
from pyrad.utils import visualization, writer


class NGPGraph(Graph):

    """Instant NGP graph"""

    def __init__(self, field_implementation="torch", intrinsics=None, camera_to_world=None, **kwargs) -> None:
        assert field_implementation in field_implementation_to_class
        self.field_implementation = field_implementation
        self.field = None
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """Set the fields."""
        # torch or tiny-cuda-nn version
        self.field = field_implementation_to_class[self.field_implementation](self.scene_bounds.aabb)

    def populate_misc_modules(self):
        # occupancy grid
        self.occupancy_grid = OccupancyGrid(aabb=self.scene_bounds.aabb)

        # samplers
        self.sampler_occupancy_grid = UniformSampler(num_samples=128, occupancy_field=self.occupancy_grid)
        # NOTE(ethan): are we sure we want the include_original flag used like this?
        # it could be easily forgotten that it's by default True...?
        self.sampler_pdf = PDFSampler(num_samples=128, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        # uniform sampling
        ray_samples_uniform = self.sampler_occupancy_grid(ray_bundle)
        field_outputs_uniform = self.field.forward(ray_samples_uniform.to_point_samples())
        weights_uniform = ray_samples_uniform.get_weights(field_outputs_uniform[FieldHeadNames.DENSITY])

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_uniform)
        field_outputs_pdf = self.field.forward(ray_samples_pdf.to_point_samples())

        # Hacky treatment of bins as points to allow us to merge uniform and pdf.
        ts_uniform = (ray_samples_uniform.bin_starts + ray_samples_uniform.bin_ends) / 2.0
        ts_pdf = (ray_samples_pdf.bin_starts + ray_samples_pdf.bin_ends) / 2.0
        ts, indices = torch.sort(torch.cat([ts_uniform, ts_pdf], -1), -1)
        ray_samples = ray_bundle.get_ray_samples(bin_starts=ts, bin_ends=ts)
        field_outputs = {}
        for fo_name, _ in field_outputs_pdf.items():
            fo_uniform = field_outputs_uniform[fo_name]
            fo_pdf = field_outputs_pdf[fo_name]
            fo_uniform_pdf = torch.cat([fo_uniform, fo_pdf], 1)
            index = indices.view(fo_uniform_pdf[..., :1].shape)
            index = index.expand(-1, -1, fo_uniform_pdf.shape[-1])  # TODO: don't hardcode this
            field_outputs[fo_name] = torch.gather(fo_uniform_pdf, dim=1, index=index)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
        )
        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples)

        densities_occupancy_grid = self.occupancy_grid.get_densities(ray_samples.frustums.get_positions())
        weights_occupancy_grid = ray_samples.get_weights(densities_occupancy_grid)
        depth_occupancy_grid = self.renderer_depth(weights_occupancy_grid, ray_samples)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "depth_occupancy_grid": depth_occupancy_grid,
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        device = self.get_device()
        image = batch["image"].to(device)
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb = outputs["rgb"]
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        writer.put_image(name=f"img/image_idx_{image_idx}", image=combined_rgb, step=step)
        writer.put_image(name=f"accumulation/image_idx_{image_idx}", image=combined_acc, step=step)
        writer.put_image(name=f"depth/image_idx_{image_idx}", image=combined_depth, step=step)

        depth = visualization.apply_depth_colormap(outputs["depth_occupancy_grid"])
        writer.put_image(name=f"depth_occupancy_grid/image_idx_{image_idx}", image=combined_depth, step=step)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        writer.put_scalar(name=f"psnr/val_{image_idx}-fine", scalar=float(psnr), step=step)
        writer.put_scalar(name=f"ssim/val_{image_idx}", scalar=float(ssim), step=step)
        writer.put_scalar(name=f"lpips/val_{image_idx}", scalar=float(lpips), step=step)

        writer.put_scalar(name=writer.EventName.CURR_TEST_PSNR, scalar=float(psnr), step=step)

        return psnr.item()
