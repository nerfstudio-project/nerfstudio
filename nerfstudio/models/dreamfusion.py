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
DreamFusion implementation.
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
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

# from nerfstudio.field_components.encodings import NeRFEncoding,
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.dreamfusion_field import DreamFusionField
from nerfstudio.model_components.losses import (
    MSELoss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, SphereCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, math, misc


@dataclass
class DreamFusionModelConfig(ModelConfig):
    """DreamFusion model config"""

    _target: Type = field(default_factory=lambda: DreamFusionModel)
    """target class to instantiate"""

    num_samples: int = 256
    """Number of samples in field evaluation"""
    orientation_loss_mult: float = 0.01
    """Orientation loss multipier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    random_light_source: bool = True
    """Randomizes light source per output."""
    initialize_density: bool = True
    """Initialize density in center of scene."""
    init_density_strength: float = 50.0
    """Initial strength of center density"""
    sphere_collider: bool = True
    """Use spherical collider instead of box"""


class DreamFusionModel(Model):
    """DreamFusionModel Model

    Args:
        config: DreamFusion configuration to instantiate model
    """

    config: DreamFusionModelConfig

    def __init__(
        self,
        config: DreamFusionModelConfig,
        **kwargs,
    ) -> None:
        self.num_samples = config.num_samples
        self.initialize_density = config.initialize_density
        self.train_normals = False
        self.train_shaded = False
        self.density_strength = 1.0
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:

        # the callback that we want to run every X iterations after the training iteration
        def taper_density(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.density_strength -= 0.2

        def start_training_normals(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.train_normals = True

        def start_shaded_training(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.train_shaded = True

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(200, 400, 600, 800, 1000),
                func=taper_density,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(1000,),
                func=start_training_normals,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(1000,),
                func=start_shaded_training,
                args=[self, training_callback_attributes],
            ),
        ]
        return callbacks

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields

        self.field = DreamFusionField(self.scene_box.aabb)

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_samples, single_jitter=True)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        # colliders
        if self.config.sphere_collider:
            self.collider = SphereCollider(torch.Tensor([0, 0, 0]), 1.0)
        else:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        background_rgb = self.field.get_background_rgb(ray_bundle)
        # ray_bundle = self.sphere_collider(ray_bundle)
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        field_outputs = self.field(ray_samples_uniform, compute_normals=True)

        density = field_outputs[FieldHeadNames.DENSITY]
        if self.initialize_density and self.training:
            pos = ray_samples_uniform.frustums.get_positions()
            density_blob = (
                self.config.init_density_strength
                * self.density_strength
                * torch.exp(-torch.norm(pos, dim=-1) / (2 * 0.04))[..., None]
            )
            density += density_blob

        weights = ray_samples_uniform.get_weights(density)

        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples_uniform)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        # rgb = torch.clamp(rgb, 0.0, 1.0)

        accum_mask = torch.clamp((torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0)
        accum_mask_inv = 1.0 - accum_mask

        background = accum_mask_inv * background_rgb

        outputs = {
            "rgb": rgb,
            "background_rgb": background_rgb,
            "background": background,
            "accumulation": accum_mask,
            "depth": depth,
        }

        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)  # .detach()

        # lambertian shading
        if self.config.random_light_source and self.training:  # and if training
            light_d = ray_bundle.origins[0] + torch.randn(3, dtype=torch.float).to(normals)
        else:
            light_d = ray_bundle.origins[0]
        light_d = math.safe_normalize(light_d)

        if self.train_shaded and np.random.random_sample() > 0.75:
            ratio = 0.1
        else:
            ratio = 0.0
        # with torch.no_grad():
        lambertian = ratio + (1 - ratio) * (pred_normals.detach() @ light_d).clamp(min=0)
        shaded = lambertian.unsqueeze(-1).repeat(1, 3)
        shaded_albedo = rgb * lambertian.unsqueeze(-1)

        outputs["normals"] = normals
        outputs["pred_normals"] = pred_normals
        outputs["shaded"] = accum_mask * shaded + torch.ones_like(shaded) * accum_mask_inv
        outputs["shaded_albedo"] = shaded_albedo
        outputs["render"] = accum_mask * rgb + background

        if ratio > 0:
            if np.random.random_sample() > 0.5:
                outputs["train_output"] = shaded
            else:
                outputs["train_output"] = accum_mask * shaded_albedo + background
        else:
            outputs["train_output"] = outputs["render"]

        # if self.train_normals:

        outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
        )

        outputs["rendered_pred_normal_loss"] = pred_normal_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS].detach(),
            field_outputs[FieldHeadNames.PRED_NORMALS],
        )

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.

        loss_dict = {}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        # if self.train_normals:
        # orientation loss for computed normals
        loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
            outputs["rendered_orientation_loss"]
        )

        # ground truth supervision for normals
        loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
            outputs["rendered_pred_normal_loss"]
        )
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
