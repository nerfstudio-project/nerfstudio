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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.fields.modules.encoding import NeRFEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.nerf_field import NeRFField
from nerfactory.models.base import Model
from nerfactory.models.modules.ray_sampler import PDFSampler, UniformSampler
from nerfactory.models.modules.scene_colliders import NearFarCollider
from nerfactory.optimizers.loss import MSELoss
from nerfactory.renderers.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfactory.utils import colors, misc, visualization, writer
from nerfactory.utils.callbacks import Callback


class NeRFModel(Model):
    """Vanilla NeRF model

    Args:
        near_plane: Where to start sampling points. Defaults to 2.0.
        far_plane: Where to stop sampling points. Defaults to 6.0.
        num_coarse_samples: Number of samples in coarse field evaluation. Defaults to 64,
        num_importance_samples: Number of samples in fine field evaluation. Defaults to 64,
        enable_density_field: Whether to create a density field to filter samples. Defaults to False.
        density_field_config: Configuration of density field. Defaults to None.
    """

    def __init__(
        self,
        config: cfg.ModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def get_training_callbacks(self) -> List[Callback]:
        if self.field_coarse is None:
            raise ValueError("populate fields must be called before get_training_callbacks.")

        callbacks = []
        if self.density_field is not None:
            callbacks = [
                Callback(
                    update_every_num_iters=self.density_field.update_every_num_iters,
                    func=self.density_field.update_density_grid,
                    density_eval_func=self.field_coarse.density_fn,
                )
            ]
        return callbacks  # type: ignore

    def populate_fields(self):
        """Set the fields."""

        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

    def populate_misc_modules(self):
        # samplers
        self.sampler_uniform = UniformSampler(
            num_samples=self.config.num_coarse_samples, density_field=self.density_field
        )
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, density_field=self.density_field)

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

        # colliders
        if self.config.enable_collider:
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict, loss_coefficients) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)

        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, loss_coefficients)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = visualization.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = visualization.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = visualization.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = visualization.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        writer.put_image(name=f"img/image_idx_{image_idx}", image=combined_rgb, step=step)
        writer.put_image(name=f"accumulation/image_idx_{image_idx}", image=combined_acc, step=step)
        writer.put_image(name=f"depth/image_idx_{image_idx}", image=combined_depth, step=step)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        writer.put_scalar(name=f"psnr/val_{image_idx}-coarse", scalar=float(coarse_psnr), step=step)
        writer.put_scalar(name=f"psnr/val_{image_idx}-fine", scalar=float(fine_psnr), step=step)
        writer.put_scalar(name=f"ssim/val_{image_idx}", scalar=float(fine_ssim), step=step)  # type: ignore
        writer.put_scalar(name=f"lpips/val_{image_idx}", scalar=float(fine_lpips), step=step)

        writer.put_scalar(name=writer.EventName.CURR_TEST_PSNR, scalar=float(fine_psnr), step=step)

        return fine_psnr.item()
