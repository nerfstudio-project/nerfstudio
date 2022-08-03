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
TensorRF implementation.
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfactory.cameras.rays import RayBundle
from nerfactory.fields.modules.encoding import TensorVMEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.tensorf_field import TensoRFField
from nerfactory.graphs.base import Graph
from nerfactory.graphs.modules.ray_sampler import PDFSampler, UniformSampler
from nerfactory.optimizers.loss import MSELoss
from nerfactory.renderers.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfactory.utils import colors, misc, visualization, writer
from nerfactory.utils.callbacks import Callback


class TensoRFGraph(Graph):
    """
    TensoRF Graph
    """

    def __init__(
        self,
        intrinsics: torch.Tensor = None,
        camera_to_world: torch.Tensor = None,
        near_plane: float = 2.0,
        far_plane: float = 6.0,
        num_coarse_samples: int = 64,
        num_importance_samples: int = 128,
        enable_density_field: bool = False,
        **kwargs,
    ) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.field = None
        self.num_coarse_samples = num_coarse_samples
        self.num_importance_samples = num_importance_samples
        super().__init__(
            intrinsics=intrinsics, camera_to_world=camera_to_world, enable_density_field=enable_density_field, **kwargs
        )

    def get_training_callbacks(self) -> List[Callback]:
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_training_callbacks")
        callbacks = [
            Callback(update_every_num_iters=1000, func=self.field.position_encoding.upsample_grid),
            Callback(update_every_num_iters=1000, func=self.field.direction_encoding.upsample_grid),
        ]

        return callbacks  # type: ignore

    def populate_fields(self):
        """Set the fields."""

        position_encoding = TensorVMEncoding(
            resolution=128, num_components=8, final_resolution=300, num_upsampling_steps=0
        )
        direction_encoding = TensorVMEncoding(
            resolution=128, num_components=24, final_resolution=300, num_upsampling_steps=0
        )

        self.field = TensoRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=2,
            base_mlp_layer_width=128,
        )

    def populate_misc_modules(self):
        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_coarse_samples, density_field=self.density_field)
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples, density_field=self.density_field)

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
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        param_groups["position_encoding"] = list(self.field.position_encoding.parameters())
        param_groups["direction_encoding"] = list(self.field.direction_encoding.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field:
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights,
        )
        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples_uniform)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict, loss_coefficients) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, loss_coefficients)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb = outputs["rgb"]
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        writer.put_image(name=f"img/image_idx_{image_idx}", image=combined_rgb, step=step)
        writer.put_image(name=f"accumulation/image_idx_{image_idx}", image=acc, step=step)
        writer.put_image(name=f"depth/image_idx_{image_idx}", image=depth, step=step)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        writer.put_scalar(name=f"psnr/val_{image_idx}", scalar=float(psnr), step=step)
        writer.put_scalar(name=f"ssim/val_{image_idx}", scalar=float(ssim), step=step)  # type: ignore
        writer.put_scalar(name=f"lpips/val_{image_idx}", scalar=float(lpips), step=step)

        writer.put_scalar(name=writer.EventName.CURR_TEST_PSNR, scalar=float(psnr), step=step)

        return psnr.item()
