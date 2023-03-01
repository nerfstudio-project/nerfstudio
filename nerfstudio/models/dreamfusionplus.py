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
from typing import Dict, List, Tuple, Type, Optional, Literal

import numpy as np
import torch
from torch.nn import Parameter
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.generative.stable_diffusion_utils import PositionalTextEmbeddings
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.dreamfusion_field import DreamFusionField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, SphereCollider
from nerfstudio.model_components.shaders import LambertianShader, NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.dreamfusion import DreamFusionModel, DreamFusionModelConfig
from nerfstudio.utils import colormaps, colors, math, misc


@dataclass
class DreamFusionPlusModelConfig(DreamFusionModelConfig):
    """DreamFusion model config"""

    _target: Type = field(default_factory=lambda: DreamFusionPlusModel)
    """target class to instantiate"""
    # prompt: str = "A high-quality photo of a tree frog on a stump"
    prompt: str = "A high quality photo of a <nicechair2>"
    # """prompt for stable dreamfusion"""


class DreamFusionPlusModel(DreamFusionModel):
    """DreamFusionModel Model

    Args:
        config: DreamFusion configuration to instantiate model
    """

    config: DreamFusionPlusModelConfig

    def __init__(
        self,
        config: DreamFusionPlusModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.

        loss_dict = {}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        if self.train_normals:
            # orientation loss for computed normals
            loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                outputs["rendered_orientation_loss"]
            )
            # ground truth supervision for normals
            loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                outputs["rendered_pred_normal_loss"]
            )
        else:
            loss_dict["orientation_loss"] = 0
            loss_dict["pred_normal_loss"] = 0

        if self.config.opacity_penalty:
            loss_dict["opacity_loss"] = self.config.opacity_loss_mult * outputs["opacity_loss"].mean()

        if self.prompt != self.cur_prompt:
            self.cur_prompt = self.prompt
            self.text_embeddings.update_prompt(
                base_prompt=self.cur_prompt,
                top_prompt=self.cur_prompt + self.top_prompt,
                side_prompt=self.cur_prompt + self.side_prompt,
                back_prompt=self.cur_prompt + self.back_prompt,
                front_prompt=self.cur_prompt + self.front_prompt,
            )

        if batch["input_image"] == True:
            loss_dict["rgb_loss"] = self.rgb_loss(batch["image"].to(self.device), outputs["rgb"])
            loss_dict['sds_loss'] = 0
        else:
            
            # raise Exception
            text_embedding = self.text_embeddings.get_text_embedding(
                vertical_angle=batch["vertical"], horizontal_angle=batch["central"]
            )                
            train_output = (
                outputs["train_output"].view(1, int(outputs["train_output"].shape[0] ** 0.5), int(outputs["train_output"].shape[0] ** 0.5), 3)
                .permute(0, 3, 1, 2)
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sds_loss = self.sd.sds_loss(
                    text_embedding.to(self.sd_device),
                    train_output.to(self.sd_device),
                    guidance_scale=int(self.guidance_scale),
                    grad_scaler=self.grad_scaler,
                )

            loss_dict["sds_loss"] = sds_loss.to(self.device)
            loss_dict["rgb_loss"] = 0


        if self.training:
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return loss_dict