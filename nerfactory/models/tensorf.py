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

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.fields.modules.encoding import TensorVMEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.nerf_field import NeRFField
from nerfactory.models.base import Model
from nerfactory.models.modules.ray_sampler import PDFSampler, UniformSampler
from nerfactory.optimizers.loss import L1Loss, MSELoss
from nerfactory.optimizers.optimizers import Optimizers
from nerfactory.renderers.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfactory.utils import colors, misc, visualization, writer
from nerfactory.utils.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)


class TensoRFModel(Model):
    """
    TensoRF Model
    """

    def __init__(
        self,
        config: cfg.TensoRFModelConfig,
        **kwargs,
    ) -> None:
        self.init_resolution = config.init_resolution
        self.upsampling_iters = config.upsampling_iters
        self.upsampling_steps = (
            np.round(
                np.exp(
                    np.linspace(
                        np.log(config.init_resolution),
                        np.log(config.final_resolution),
                        len(config.upsampling_iters) + 1,
                    )
                )
            )
            .astype("int")
            .tolist()[1:]
        )
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:

        # the callback that we want to run every X iterations after the training iteration
        def reinitialize_optimizer(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            resolution = self.upsampling_steps.pop(0)

            # upsample the position and direction grids
            # TODO(ethan): ask Brent how to get typing to work on this... the Encoding base class type
            # in NeRFField is causing the issue
            self.field.position_encoding.upsample_grid(resolution)
            self.field.direction_encoding.upsample_grid(resolution)

            # reinitialize the optimizer
            optimizers_config = training_callback_attributes.optimizers.config
            training_callback_attributes.optimizers = Optimizers(
                optimizers_config, training_callback_attributes.pipeline.get_param_groups()
            )
            # TODO(ethan): do something with the learning rate
            # we don't want to reinitialize the learning rate each time

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=self.upsampling_iters,
                func=reinitialize_optimizer,
                args=[self, training_callback_attributes],
            )
        ]
        return callbacks

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = TensorVMEncoding(
            resolution=self.init_resolution,
            num_components=96,
        )
        direction_encoding = TensorVMEncoding(
            resolution=self.init_resolution,
            num_components=96,
        )

        self.field = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=2,
            base_mlp_layer_width=128,
        )

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
        self.feature_loss = L1Loss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        param_groups["position_encoding"] = list(self.field.position_encoding.parameters())
        param_groups["direction_encoding"] = list(self.field.direction_encoding.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

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
        assert isinstance(self.field.position_encoding, TensorVMEncoding)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        plane_coef = self.field.position_encoding.plane_coef
        line_coef = self.field.position_encoding.line_coef

        plane_feature_loss = self.feature_loss(plane_coef, torch.zeros_like(plane_coef))
        line_feature_loss = self.feature_loss(line_coef, torch.zeros_like(line_coef))

        loss_dict = {"rgb_loss": rgb_loss, "feature_loss": plane_feature_loss + line_feature_loss}
        loss_dict = misc.scale_dict(loss_dict, loss_coefficients)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb = outputs["rgb"]
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
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
