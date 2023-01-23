# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import NeRFEncoding, TensorVMEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class TensoRFModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: TensoRFModel)
    """target class to instantiate"""
    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 300
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    """specifies a list of iteration step numbers to perform upsampling"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    """Loss specific weights."""
    num_samples: int = 256
    """Number of samples in field evaluation"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    appearance_dim: int = 27
    """Number of channels for color encoding"""


class TensoRFModel(Model):
    """TensoRF Model

    Args:
        config: TensoRF configuration to instantiate model
    """

    def __init__(
        self,
        config: TensoRFModelConfig,
        **kwargs,
    ) -> None:
        self.init_resolution = config.init_resolution
        self.upsampling_iters = config.upsampling_iters
        self.num_den_components = config.num_den_components
        self.num_color_components = config.num_color_components
        self.appearance_dim = config.appearance_dim
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
            index = self.upsampling_iters.index(step)
            resolution = self.upsampling_steps[index]

            # upsample the position and direction grids
            self.field.density_encoding.upsample_grid(resolution)
            self.field.color_encoding.upsample_grid(resolution)

            # reinitialize the encodings optimizer
            optimizers_config = training_callback_attributes.optimizers.config
            enc = training_callback_attributes.pipeline.get_param_groups()["encodings"]
            lr_init = optimizers_config["encodings"]["optimizer"].lr

            training_callback_attributes.optimizers.optimizers["encodings"] = optimizers_config["encodings"][
                "optimizer"
            ].setup(params=enc)
            if optimizers_config["encodings"]["scheduler"]:
                training_callback_attributes.optimizers.schedulers["encodings"] = optimizers_config["encodings"][
                    "scheduler"
                ].setup(optimizer=training_callback_attributes.optimizers.optimizers["encodings"], lr_init=lr_init)

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=self.upsampling_iters,
                func=reinitialize_optimizer,
                args=[self, training_callback_attributes],
            )
        ]
        return callbacks

    def update_to_step(self, step: int) -> None:
        if step < self.upsampling_iters[0]:
            return

        new_iters = list(self.upsampling_iters) + [step + 1]
        new_iters.sort()

        index = new_iters.index(step + 1)
        new_grid_resolution = self.upsampling_steps[index - 1]

        self.field.density_encoding.upsample_grid(new_grid_resolution)  # type: ignore
        self.field.color_encoding.upsample_grid(new_grid_resolution)  # type: ignore

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        density_encoding = TensorVMEncoding(
            resolution=self.init_resolution,
            num_components=self.num_den_components,
        )
        color_encoding = TensorVMEncoding(
            resolution=self.init_resolution,
            num_components=self.num_color_components,
        )

        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.field = TensoRFField(
            self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            appearance_dim=self.appearance_dim,
            head_mlp_num_layers=2,
            head_mlp_layer_width=128,
            use_sh=False,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples // 2, single_jitter=True)

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
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
            self.field.density_encoding.parameters()
        )

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
