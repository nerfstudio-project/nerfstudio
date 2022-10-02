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
NeRF-W (NeRF in the wild) implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encoding import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerf_field import NeRFField
from nerfstudio.fields.nerfw_field import VanillaNerfWField
from nerfstudio.model_components.loss import MSELoss
from nerfstudio.model_components.ray_sampler import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base import Model, VanillaModelConfig
from nerfstudio.utils import colors, misc, visualization


@dataclass
class NerfWModelConfig(VanillaModelConfig):
    """NerfW model config"""

    _target: Type = field(default_factory=lambda: NerfWModel)
    """target class to instantiate"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "uncertainty_loss": 1.0, "density_loss": 0.01}
    )
    """Loss specific weights."""
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation."""
    num_importance_samples: int = 64
    """Number of samples in fine field evaluation."""
    uncertainty_min: float = 0.03
    """This is added to the end of the uncertainty
    rendering operation. It's called 'beta_min' in other repos.
    This avoids calling torch.log() on a zero value, which would be undefined.
    """
    num_images: int = 10000  # TODO: don't hardcode this
    """How many images exist in the dataset."""
    appearance_embedding_dim: int = 48
    """Dimension of appearance embedding."""
    transient_embedding_dim: int = 16
    """Dimension of transient embedding."""


class NerfWModel(Model):
    """NeRF-W model

    Args:
        config: NerfW configuration to instantiate model
    """

    config: NerfWModelConfig

    def __init__(
        self,
        config: NerfWModelConfig,
        **kwargs,
    ) -> None:
        """A NeRF-W model.

        Args:
            ...
            uncertainty_min (float, optional): This is added to the end of the uncertainty
                rendering operation. It's called 'beta_min' in other repos.
                This avoids calling torch.log() on a zero value, which would be undefined.
        """
        self.field_coarse = None
        self.field_fine = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = VanillaNerfWField(
            num_images=self.config.num_images,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            transient_embedding_dim=self.config.transient_embedding_dim,
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.BLACK)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self):
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if ray_bundle.camera_indices is not None:
            # TODO(ethan): remove this check
            assert (
                torch.max(ray_bundle.camera_indices) < self.config.num_images
            ), "num_images must be greater than the largest camera index"

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)

        # fine weights
        weights_fine = ray_samples_pdf.get_weights(
            field_outputs_fine[FieldHeadNames.DENSITY] + field_outputs_fine[FieldHeadNames.TRANSIENT_DENSITY]
        )
        weights_fine_static = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        weights_fine_transient = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.TRANSIENT_DENSITY])

        # rgb
        rgb_fine_static_component = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        rgb_fine_transient_component = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.TRANSIENT_RGB],
            weights=weights_fine,
        )
        rgb_fine = rgb_fine_static_component + rgb_fine_transient_component
        rgb_fine_static = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine_static,
        )

        # density
        density_transient = field_outputs_fine[FieldHeadNames.TRANSIENT_DENSITY]

        # depth
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        depth_fine_static = self.renderer_depth(weights_fine_static, ray_samples_pdf)

        # uncertainty
        uncertainty = self.renderer_uncertainty(field_outputs_fine[FieldHeadNames.UNCERTAINTY], weights_fine_transient)
        uncertainty += self.config.uncertainty_min

        outputs = {
            "rgb_coarse": rgb_coarse,  # (num_rays, 3)
            "rgb_fine": rgb_fine,
            "rgb_fine_static": rgb_fine_static,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "depth_fine_static": depth_fine_static,
            "density_transient": density_transient,  # (num_rays, num_samples, 1)
            "uncertainty": uncertainty,  # (num_rays, 1)
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        density_transient = outputs["density_transient"]
        betas = outputs["uncertainty"]
        rgb_loss_coarse = 0.5 * ((image - rgb_coarse) ** 2).sum(-1).mean()
        rgb_loss_fine = 0.5 * (((image - rgb_fine) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        uncertainty_loss = 0.5 * (3 + torch.log(betas)).mean()
        density_loss = density_transient.mean()

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "uncertainty_loss": uncertainty_loss,
            "density_loss": density_loss,
        }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        rgb_fine_static = outputs["rgb_fine_static"]
        depth_coarse = outputs["depth_coarse"]
        depth_fine = outputs["depth_fine"]
        depth_fine_static = outputs["depth_fine_static"]
        uncertainty = outputs["uncertainty"]

        depth_coarse = visualization.apply_depth_colormap(depth_coarse)
        depth_fine = visualization.apply_depth_colormap(depth_fine)
        depth_fine_static = visualization.apply_depth_colormap(depth_fine_static)
        uncertainty = visualization.apply_depth_colormap(uncertainty)

        row0 = torch.cat([image, uncertainty, torch.ones_like(rgb_fine)], dim=-2)
        row1 = torch.cat([rgb_fine, rgb_fine_static, rgb_coarse], dim=-2)
        row2 = torch.cat([depth_fine, depth_fine_static, depth_coarse], dim=-2)
        combined_image = torch.cat([row0, row1, row2], dim=-3)

        # this doesn't really make sense but do it anyway
        fine_psnr = self.psnr(image, rgb_fine)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
        }
        images_dict = {"img": combined_image}
        if "mask" in batch:
            mask = batch["mask"].repeat(1, 1, 3)
            images_dict["mask"] = mask
        return metrics_dict, images_dict
