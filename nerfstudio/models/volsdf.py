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
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Type
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import S3IM
from nerfstudio.model_components.ray_samplers import ErrorBoundedSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.model_components.losses import monosdf_normal_loss


import torch
import torch.nn.functional as F
from torch.nn import Parameter

@dataclass
class VolSDFModelConfig(SurfaceModelConfig):
    """VolSDF Model Config"""

    _target: Type = field(default_factory=lambda: VolSDFModel)
    num_samples: int = 64
    """Number of samples after error bounded sampling"""
    num_samples_eval: int = 128
    """Number of samples per iteration used in error bounded sampling"""
    num_samples_extra: int = 32
    """Number of uniformly sampled points for training"""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""

    # from SDFStudio base surface model
    s3im_loss_mult: float = 0.0
    """S3IM loss multiplier."""
    s3im_kernel_size: int = 4
    """S3IM kernel size."""
    s3im_stride: int = 4
    """S3IM stride."""
    s3im_repeat_time: int = 10
    """S3IM repeat time."""
    s3im_patch_height: int = 32
    """S3IM virtual patch height."""


class VolSDFModel(SurfaceModel):
    """VolSDF model

    Args:
        config: VolSDF configuration to instantiate model
    """

    config: VolSDFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = ErrorBoundedSampler(
            num_samples=self.config.num_samples,
            num_samples_eval=self.config.num_samples_eval,
            num_samples_extra=self.config.num_samples_extra,
        )
        self.s3im_loss = S3IM(s3im_kernel_size=self.config.s3im_kernel_size, 
                              s3im_stride=self.config.s3im_stride, 
                              s3im_repeat_time=self.config.s3im_repeat_time, 
                              s3im_patch_height=self.config.s3im_patch_height)
        
        # (optional) camera optimizer
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )


    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples, eik_points = self.sampler(
            ray_bundle, density_fn=self.field.laplace_density, sdf_fn=self.field.get_sdf
        )
        field_outputs = self.field(ray_samples)
        weights, transmittance = ray_samples.get_weights_and_transmittance(field_outputs[FieldHeadNames.DENSITY])
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "eik_points": eik_points,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["beta"] = self.field.laplace_density.get_beta().item()
            metrics_dict["alpha"] = 1.0 / self.field.laplace_density.get_beta().item()

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(image, pred_image)
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult
             # s3im loss
            if self.config.s3im_loss_mult > 0:
                loss_dict["s3im_loss"] = self.s3im_loss(image, outputs["rgb"]) * self.config.s3im_loss_mult
            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

            # monocular normal loss
            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                normal_gt = batch["normal"].to(self.device)
                normal_pred = outputs["normal"]
                loss_dict["normal_loss"] = (
                    monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
                )

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                    * self.config.mono_depth_loss_mult
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups
    
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        # just add the camera optimizer to the ray bundle
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        return super().get_outputs(ray_bundle)
    