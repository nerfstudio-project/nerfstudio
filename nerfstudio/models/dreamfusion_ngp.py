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

from nerfstudio.fields.dreamngp_field import DreamNGPField
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.dreamfusion_field import DreamFusionField
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
    ProposalNetworkSampler,
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
class DreamfusionNGPModelConfig(DreamFusionModelConfig):
    """DreamFusion model config"""

    _target: Type = field(default_factory=lambda: DreamfusionNGPModel)
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
    
    contraction_type: ContractionType = ContractionType.UN_BOUNDED_SPHERE
    """Contraction type used for spatial deformation of the field."""

    grid_resolution: int = 64
    """Resolution of the grid used for the field."""
    # max_res: int = 2048
    max_res = 256
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    cone_angle: float = 0.0
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: float = 0.01
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    max_num_samples_per_ray: int = 24
    

    start_normals_training: int = 1000
    """Start training normals after this many iterations"""
    start_lambertian_training: int = 1000
    """start training with lambertian shading after this many iterations"""
    opacity_penalty: bool = False
    """enables penalty to encourage sparse weights (penalizing for uniform density along ray)"""
    opacity_loss_mult: float = 1
    """scale for opacity penalty"""
    


class DreamfusionNGPModel(DreamFusionModel):
    """DreamEmbeddingModel Model

    Args:
        config: DreamFusion configuration to instantiate model
    """

    config: DreamfusionNGPModel

    def __init__(
        self,
        config: DreamfusionNGPModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        
        # setting up fields
        self.field = DreamNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=False,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
        )
        self.initialize_density=self.config.initialize_density
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        
        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # samplers
        self.sampler = VolumetricSampler(
            scene_aabb=self.scene_box.aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

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
        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        def start_training_normals(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.train_normals = True

        def taper_density(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            self.density_strength = np.interp(step, self.config.taper_range, self.config.taper_strength)

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),            
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(self.config.start_normals_training,),
                func=start_training_normals,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=taper_density,
                update_every_num_iters=1,
                args=[self, training_callback_attributes],
            ),
        ]
        
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )
        # field_outputs = self.field(ray_samples, compute_normals=True)
        field_outputs = self.field(ray_samples, compute_normals=False)

        density = field_outputs[FieldHeadNames.DENSITY]

        if self.initialize_density:
            pos = ray_samples.frustums.get_positions()
            density_blob = (-0.05 * torch.exp(5 * torch.norm(pos, dim=-1)) + 0.5)[..., None]
            density = torch.max(density + density_blob, torch.tensor([0.], device=self.device))

        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=density,
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        background_rgb = self.field.get_background_rgb(ray_bundle)

        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0

        accum_mask = torch.clamp((torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0)
        accum_mask_inv = 1.0 - accum_mask

        background = accum_mask_inv * background_rgb

        outputs = {
            "rgb_only": rgb,
            "background_rgb": background_rgb,
            "background": background,
            "accumulation": accum_mask,
            "depth": depth,
            "rgb": accum_mask * rgb + background
        }

        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        outputs["normals"] = self.shader_normals(normals, weights=accum_mask)

        # lambertian shading
        if self.config.random_light_source:  # and self.training:
            light_d = ray_bundle.origins[0] + torch.randn(3, dtype=torch.float).to(normals)
        else:
            light_d = ray_bundle.origins[0]
        light_d = math.safe_normalize(light_d)

        if (self.train_shaded and np.random.random_sample() > 0.75) or not self.training:
            shading_weight = 0.9
        else:
            shading_weight = 0.0

        shaded, shaded_albedo = self.shader_lambertian(
            rgb=rgb, normals=normals, light_direction=light_d, shading_weight=shading_weight, detach_normals=False
        )

        outputs["shaded"] = accum_mask * shaded

        if shading_weight > 0:
            samp = np.random.random_sample()
            if samp > 0.5 and not self.training:
                outputs["train_output"] = outputs["shaded"]
            elif samp < 0.2 and self.random_background:
                rand_bg = torch.ones_like(background) * torch.rand(3, device=self.device)
                outputs["train_output"] = accum_mask * shaded_albedo + rand_bg * accum_mask_inv
            else:
                outputs["train_output"] = accum_mask * shaded_albedo + background
        else:
            outputs["train_output"] = outputs["rgb"]

        # print(outputs["rgb"].shape)
        # outputs["train_output"] = outputs["rgb"]


        outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS],
            ray_bundle.directions,
        )

        assert weights.shape[-1] == 1
        if self.config.opacity_penalty:
            outputs["opacity_loss"] = torch.sqrt(torch.sum(weights, dim=-2) ** 2 + 0.01) * self.config.opacity_loss_mult

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.

        loss_dict = {}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        if self.train_normals:
            # orientation loss for computed normals
            loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                outputs["rendered_orientation_loss"]
            )
        else:
            loss_dict["orientation_loss"] = 0

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
    