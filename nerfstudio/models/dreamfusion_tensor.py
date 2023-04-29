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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import nerfacc
import torch
import numpy as np
import torch.nn.functional as F
from nerfacc import ContractionType
from torch.nn import Parameter

from typing_extensions import Literal
from nerfstudio.models.dreamfusion import DreamFusionModel, DreamFusionModelConfig

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    TensorCPEncoding,
    TensorVMEncoding,
    TriplaneEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.dreamtensor_field import DreamFusionTensorField
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.generative.stable_diffusion_utils import PositionalTextEmbeddings
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    PDFSampler,
    UniformSampler,
)
from nerfstudio.model_components.ray_samplers import VolumetricSampler

from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, SphereCollider
from nerfstudio.model_components.shaders import LambertianShader, NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, math, misc


@dataclass
class DreamfusionTensorModelConfig(DreamFusionModelConfig):
    """DreamFusion model config"""

    _target: Type = field(default_factory=lambda: DreamfusionTensorModel)
    """target class to instantiate"""
    # prompt: str = "A high quality zoomed out photo of a teddy bear"
    prompt: str = "A high-quality photo of a pineapple"
    """prompt for stable dreamfusion"""

    orientation_loss_mult: float = 0.0001
    """Orientation loss multipier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    random_light_source: bool = True
    """Randomizes light source per output."""
    initialize_density: bool = True
    """Initialize density in center of scene."""
    taper_range: Tuple[int, int] = (0, 1000)
    """Range of step values for the density tapering"""
    taper_strength: Tuple[float, float] = (1.0, 0.0)
    """Strength schedule of center density"""
    sphere_collider: bool = True
    """Use spherical collider instead of box"""
    target_transmittance_start: float = 0.4
    """target transmittance for opacity penalty. This is the percent of the scene that is
    background when rendered at the start of training"""
    target_transmittance_end: float = 0.7
    """target transmittance for opacity penalty. This is the percent of the scene that is
    background when rendered at the end of training"""
    transmittance_end_schedule: int = 1500
    """number of iterations to reach target_transmittance_end"""

    init_resolution: int = 300
    """initial render resolution"""
    final_resolution: int = 300
    """final render resolution"""
    # upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    upsampling_iters: Tuple[int, ...] = (500, 750, 1000)
    """specifies a list of iteration step numbers to perform upsampling"""
    num_samples: int = 100
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    appearance_dim: int = 27
    """Number of channels for color encoding"""
    tensorf_encoding: Literal["triplane", "vm", "cp"] = "vm"

    start_normals_training: int = 1000
    """Start training normals after this many iterations"""
    start_lambertian_training: int = 1000
    """start training with lambertian shading after this many iterations"""
    opacity_penalty: bool = False
    """enables penalty to encourage sparse weights (penalizing for uniform density along ray)"""
    opacity_loss_mult: float = 1
    """scale for opacity penalty"""
    


class DreamfusionTensorModel(DreamFusionModel):
    """DreamTensorModel Model

    Args:
        config: DreamFusion configuration to instantiate model
    """

    config: DreamfusionTensorModel

    def __init__(
        self,
        config: DreamfusionTensorModelConfig,
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

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        if self.config.tensorf_encoding == "vm":
            density_encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorVMEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "cp":
            density_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "triplane":
            density_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )

        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.field = DreamFusionTensorField(
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
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.shader_lambertian = LambertianShader()
        self.shader_normals = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # colliders
        # if self.config.sphere_collider:
        #     self.collider = SphereCollider(torch.Tensor([0, 0, 0]), 1.0)
        # # else:
        # #     self.collider = AABBBoxCollider(scene_box=self.scene_box)
        self.collider = AABBBoxCollider(scene_box=self.scene_box)
            
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def taper_density(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.density_strength = np.interp(step, self.config.taper_range, self.config.taper_strength)

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=taper_density,
                update_every_num_iters=1,
                args=[self, training_callback_attributes],
            ),
        ]
        
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
            + list(self.field.mlp_background_color.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
            self.field.density_encoding.parameters()
        )
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle):
        background_rgb = self.field.get_background_rgb(ray_bundle)

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

        density = field_outputs_fine[FieldHeadNames.DENSITY]

        if self.initialize_density:
            pos = ray_samples_pdf.frustums.get_positions()
            density_blob = (-0.05 * torch.exp(5 * torch.norm(pos, dim=-1)) + 1.0)[..., None]
            density = torch.max(density + density_blob, torch.tensor([0.], device=self.device))

        weights_fine = ray_samples_pdf.get_weights(density)

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device, dtype=rgb.dtype), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        accum_mask = torch.clamp((torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0)
        accum_mask_inv = 1.0 - accum_mask
        background = accum_mask_inv * background_rgb

        outputs = {
            "rgb_only": rgb,
            "background_rgb": background_rgb,
            "background": background,
            "accumulation": accum_mask,
            "depth": depth,
            "train_output": accum_mask * rgb + background
        } 
        return outputs


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.

        loss_dict = {}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)

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

        text_embedding = self.text_embeddings.get_text_embedding(
            vertical_angle=batch["vertical"], horizontal_angle=batch["central"]
        )

        train_output = (
            outputs["train_output"]
            .view(1, int(outputs["train_output"].shape[0] ** 0.5), int(outputs["train_output"].shape[0] ** 0.5), 3)
            .permute(0, 3, 1, 2)
        )

        sds_loss = self._sd.sds_loss(
            text_embedding.to(self.sd_device),
            train_output.to(self.sd_device),
            guidance_scale=int(self.guidance_scale),
            grad_scaler=self.grad_scaler,
        )

        loss_dict["sds_loss"] = sds_loss.to(self.device)
        return loss_dict
    