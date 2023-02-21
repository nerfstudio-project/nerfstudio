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
    prompt: str = "A high-quality photo of a tree frog on a stump"
    """prompt for stable dreamfusion"""

    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    random_light_source: bool = True
    """Randomizes light source per output."""
    initialize_density: bool = True
    """Initialize density in center of scene."""
    init_density_strength: float = 10
    """Initial strength of center density"""
    sphere_collider: bool = True
    """Use spherical collider instead of box"""
    random_background: bool = True
    """Randomly choose between using background mlp and random color for background"""
    target_transmittance_start: float = 0.4
    """target transmittance for opacity penalty. This is the percent of the scene that is
    background when rendered at the start of training"""
    target_transmittance_end: float = 0.7
    """target transmittance for opacity penalty. This is the percent of the scene that is
    background when rendered at the end of training"""
    transmittance_end_schedule: int = 1500
    """number of iterations to reach target_transmittance_end"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 500
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 1.0
    """Distortion loss multiplier."""
    start_normals_training: int = 1000
    """Start training normals after this many iterations"""
    start_lambertian_training: int = 1000
    """start training with lambertian shading after this many iterations"""
    opacity_penalty: bool = True
    """enables penalty to encourage sparse weights (penalizing for uniform density along ray)"""
    opacity_loss_mult: float = 1
    """scale for opacity penalty"""
    max_res: int = 256
    """Maximum resolution of the density field."""

    location_based_prompting: bool = True
    """enables location based prompting"""
    interpolated_prompting: bool = False
    """enables interpolated location prompting"""
    positional_prompting: Literal["discrete", "interpolated", "base"] = "discrete"
    """ how to incorporate position into prompt"""
    top_prompt: str = ", overhead view"
    """appended to prompt for overhead view"""
    side_prompt: str = ", side view"
    """appended to prompt for side view"""
    front_prompt: str = ", front view"
    """appended to prompt for front view"""
    back_prompt: str = ", back view"
    """appended to prompt for back view"""
    guidance_scale: float = 100
    """guidance scale for sds loss"""
    stablediffusion_device: Optional[str] = None
    """device for stable diffusion"""
    sd_version: str = "1-5"


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
            text_embedding = self.text_embeddings.get_text_embedding(
                vertical_angle=batch["vertical"], horizonal_angle=batch["central"]
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