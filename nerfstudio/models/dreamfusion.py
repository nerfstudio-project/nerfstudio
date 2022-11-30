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

# from nerfstudio.field_components.encodings import NeRFEncoding,
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.dreamfusion_field import DreamFusionField
from nerfstudio.model_components.losses import (
    MSELoss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, math, misc


@dataclass
class DreamFusionModelConfig(ModelConfig):
    """DreamFusion model config"""

    _target: Type = field(default_factory=lambda: DreamFusionModel)
    """target class to instantiate"""

    num_samples: int = 256
    """Number of samples in field evaluation"""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""


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
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields

        self.field = DreamFusionField(self.scene_box.aabb)

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.num_samples // 2, single_jitter=True)

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
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        field_outputs = self.field(ray_samples_uniform, compute_normals=True)
        weights = ray_samples_uniform.get_weights(field_outputs[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples_uniform)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        # lambertian shading
        light_d = ray_bundle.origins[0] + torch.randn(3, dtype=torch.float).to(normals)
        light_d = math.safe_normalize(light_d)

        ratio = 0.1
        lambertian = ratio + (1 - ratio) * (pred_normals @ light_d).clamp(min=0)  # [N,]

        shaded = lambertian.unsqueeze(-1).repeat(1, 3)
        shaded_albedo = rgb * lambertian.unsqueeze(-1)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normals": normals,
            "pred_normals": pred_normals,
            "shaded": shaded,
            "shaded_albedo": shaded_albedo,
        }

        if self.training:
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
        # TODO: modify to not take in data images, need to get diffusion loss here
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        if self.training:
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
