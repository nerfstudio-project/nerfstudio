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
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure

from pyrad.fields.modules.field_heads import FieldHeadNames
from pyrad.fields.instant_ngp_field import field_implementation_to_class
from pyrad.graphs.base import Graph
from pyrad.optimizers.loss import MSELoss
from pyrad.fields.occupancy_fields.occupancy_grid import OccupancyGrid
from pyrad.renderers.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from pyrad.graphs.modules.ray_sampler import UniformSampler
from pyrad.utils import colors
from pyrad.cameras.rays import RayBundle
from pyrad.utils import visualization, writer
from pyrad.utils.callbacks import Callback


class NGPGraph(Graph):

    """Instant NGP graph"""

    def __init__(self, field_implementation="torch", intrinsics=None, camera_to_world=None, **kwargs) -> None:
        assert field_implementation in field_implementation_to_class
        self.field_implementation = field_implementation
        self.field = None
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def register_callbacks(self) -> None:
        """defining callbacks to run after every training iteration"""
        self.callbacks = [
            Callback(
                self.occupancy_grid.update_every_num_iters,
                self.occupancy_grid.update_occupancy_grid,
                density_fn=self.field.density_fn,
            )
        ]

    def populate_fields(self):
        """Set the fields."""
        # torch or tiny-cuda-nn version
        self.field = field_implementation_to_class[self.field_implementation](self.scene_bounds.aabb)

    def populate_misc_modules(self):
        # occupancy grid
        self.occupancy_grid = OccupancyGrid(aabb=self.scene_bounds.aabb)

        # samplers
        self.sampler_occupancy_grid = UniformSampler(num_samples=128)

        # TODO stabalize occupancy grid.
        # self.sampler_occupancy_grid = UniformSampler(num_samples=128, occupancy_field=self.occupancy_grid)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        # uniform sampling
        ray_samples = self.sampler_occupancy_grid(ray_bundle)
        field_outputs = self.field.forward(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

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

    def process_outputs_as_images(self, outputs):  # pylint:disable=no-self-use
        """Do preprocessing to make images valid"""
        # TODO: make log_test_image_outputs use this directly
        # TODO: implement across all the different graph implementations
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        combined_acc = torch.cat([acc], dim=1)
        outputs["accumulation"] = combined_acc
        combined_depth = torch.cat([depth], dim=1)
        outputs["depth"] = combined_depth
        depth = visualization.apply_depth_colormap(outputs["depth_occupancy_grid"])
        outputs["depth_occupancy_grid"] = combined_depth

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

        occupancy_depth = visualization.apply_depth_colormap(outputs["depth_occupancy_grid"])
        writer.put_image(name=f"depth_occupancy_grid/image_idx_{image_idx}", image=occupancy_depth, step=step)

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
