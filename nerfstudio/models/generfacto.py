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
Nerfstudio's Text to 3D model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.generfacto_field import GenerfactoField
from nerfstudio.generative.deepfloyd import DeepFloyd
from nerfstudio.generative.positional_text_embeddings import PositionalTextEmbeddings
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss, orientation_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, SphereCollider
from nerfstudio.model_components.shaders import LambertianShader, NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, math, misc


@dataclass
class GenerfactoModelConfig(ModelConfig):
    """Generfacto model config"""

    _target: Type = field(default_factory=lambda: GenerfactoModel)
    """target class to instantiate"""
    prompt: str = "a high quality photo of a ripe pineapple"
    """prompt for stable dreamfusion"""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    orientation_loss_mult: Tuple[float, float] = (0.001, 10.0)
    """Orientation loss multipier on computed normals."""
    orientation_loss_mult_range: Tuple[int, int] = (0, 15000)
    """number of iterations to reach last orientation_loss_mult value"""
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
    positional_prompting: Literal["discrete", "interpolated", "off"] = "discrete"
    """ how to incorporate position into prompt"""
    top_prompt: str = ", overhead view"
    """appended to prompt for overhead view"""
    side_prompt: str = ", side view"
    """appended to prompt for side view"""
    front_prompt: str = ", front view"
    """appended to prompt for front view"""
    back_prompt: str = ", back view"
    """appended to prompt for back view"""
    guidance_scale: float = 20
    """guidance scale for sds loss"""
    diffusion_device: Optional[str] = None
    """device for diffusion model"""
    diffusion_model: Literal["stablediffusion", "deepfloyd"] = "deepfloyd"
    """diffusion model for SDS loss"""
    sd_version: str = "1-5"
    """model version when using stable diffusion"""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""


class GenerfactoModel(Model):
    """Generfacto Model

    Args:
        config: Generfacto configuration to instantiate model
    """

    config: GenerfactoModelConfig

    def __init__(
        self,
        config: GenerfactoModelConfig,
        **kwargs,
    ) -> None:
        self.prompt = config.prompt
        self.cur_prompt = config.prompt
        self.sd_version = config.sd_version
        self.initialize_density = config.initialize_density
        self.train_normals = False
        self.train_shaded = False
        self.random_background = config.random_background
        self.density_strength = 1.0
        self.target_transmittance = config.target_transmittance_start
        self.grad_scaler = kwargs["grad_scaler"]

        self.guidance_scale = config.guidance_scale
        self.top_prompt = config.top_prompt
        self.side_prompt = config.side_prompt
        self.back_prompt = config.back_prompt
        self.front_prompt = config.front_prompt

        self.diffusion_device = (
            torch.device(kwargs["device"]) if config.diffusion_device is None else torch.device(config.diffusion_device)
        )

        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        if self.config.diffusion_model == "stablediffusion":
            self._diffusion_model = StableDiffusion(self.diffusion_device, version=self.sd_version)
        elif self.config.diffusion_model == "deepfloyd":
            self._diffusion_model = DeepFloyd(self.diffusion_device)

        self.text_embeddings = PositionalTextEmbeddings(
            base_prompt=self.cur_prompt,
            top_prompt=self.cur_prompt + self.top_prompt,
            side_prompt=self.cur_prompt + self.side_prompt,
            back_prompt=self.cur_prompt + self.back_prompt,
            front_prompt=self.cur_prompt + self.front_prompt,
            diffusion_model=self._diffusion_model,
            positional_prompting=self.config.positional_prompting,
        )

        # setting up fields
        self.field = GenerfactoField(self.scene_box.aabb, max_res=self.config.max_res)

        # samplers
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()

        for i in range(num_prop_nets):
            prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
            network = HashMLPDensityField(
                self.scene_box.aabb, **prop_net_args, implementation=self.config.implementation
            )
            self.proposal_networks.append(network)
        self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=UniformSampler(single_jitter=self.config.use_single_jitter),
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.shader_lambertian = LambertianShader()
        self.shader_normals = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # colliders
        if self.config.sphere_collider:
            self.collider = SphereCollider(torch.Tensor([0, 0, 0]), 1.0)
        else:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # the callback that we want to run every X iterations after the training iteration
        def taper_density(
            self,
            training_callback_attributes: TrainingCallbackAttributes,
            step: int,  # pylint: disable=unused-argument
        ):
            self.density_strength = np.interp(step, self.config.taper_range, self.config.taper_strength)

        def start_training_normals(
            self,
            training_callback_attributes: TrainingCallbackAttributes,
            step: int,  # pylint: disable=unused-argument
        ):
            self.train_normals = True

        def start_shaded_training(
            self,
            training_callback_attributes: TrainingCallbackAttributes,
            step: int,  # pylint: disable=unused-argument
        ):
            self.train_shaded = True

        def update_orientation_loss_mult(
            self,
            training_callback_attributes: TrainingCallbackAttributes,
            step: int,  # pylint: disable=unused-argument
        ):
            if step <= self.config.start_normals_training:
                self.orientation_loss_mult = 0
            else:
                self.orientation_loss_mult = np.interp(
                    step,
                    self.config.orientation_loss_mult_range,
                    self.config.orientation_loss_mult,
                )

        # anneal the weights of the proposal network before doing PDF sampling
        def set_anneal(step):
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / self.config.proposal_weights_anneal_max_num_iters, 0, 1)

            def bias(x, b):
                return b * x / ((b - 1) * x + 1)

            anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                func=taper_density,
                update_every_num_iters=1,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(self.config.start_normals_training,),
                func=start_training_normals,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(self.config.start_lambertian_training,),
                func=start_shaded_training,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_orientation_loss_mult,
                args=[self, training_callback_attributes],
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.proposal_sampler.step_cb,
            ),
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_anneal,
            ),
        ]
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):  # pylint: disable=too-many-statements
        # uniform sampling
        background_rgb = self.field.get_background_rgb(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=True)
        density = field_outputs[FieldHeadNames.DENSITY]

        if self.initialize_density:
            pos = ray_samples.frustums.get_positions()
            density_blob = self.density_strength * (-torch.exp(torch.norm(pos, dim=-1) / 0.4) + 2)[..., None]
            density = torch.max(density + density_blob, torch.tensor([0.0], device=self.device))

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        accum_mask = torch.clamp((torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0)
        accum_mask_inv = 1.0 - accum_mask

        background = accum_mask_inv * background_rgb

        outputs = {
            "rgb_only": rgb,
            "background_rgb": background_rgb,
            "background": background,
            "accumulation": accum_mask,
            "depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)

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

        shaded, shaded_albedo = accum_mask * shaded, accum_mask * shaded_albedo

        outputs["normals"] = self.shader_normals(normals, weights=accum_mask)
        outputs["shaded"] = shaded
        outputs["other_train_output"] = shaded_albedo + background
        outputs["shaded_albedo"] = shaded_albedo
        outputs["rgb"] = accum_mask * rgb + background

        # while training 50% of the time use a random background
        if np.random.random_sample() < 0.5 and self.random_background and self.training:
            background = torch.ones_like(background) * torch.rand(3, device=self.device) * accum_mask_inv

        if shading_weight > 0:
            samp = np.random.random_sample()
            if samp > 0.5:
                outputs["train_output"] = outputs["shaded"]
            else:
                outputs["train_output"] = shaded_albedo + background
        else:
            outputs["train_output"] = accum_mask * rgb + background

        outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
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
            loss_dict["orientation_loss"] = self.orientation_loss_mult * torch.mean(
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

        sds_loss = self._diffusion_model.sds_loss(
            text_embedding.to(self.diffusion_device),
            train_output.to(self.diffusion_device),
            guidance_scale=int(self.guidance_scale),
            grad_scaler=self.grad_scaler,
        )

        loss_dict["sds_loss"] = sds_loss.to(self.device)

        if self.training:
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        prop_depth_0 = colormaps.apply_depth_colormap(
            outputs["prop_depth_0"],
            accumulation=outputs["accumulation"],
        )
        prop_depth_1 = colormaps.apply_depth_colormap(
            outputs["prop_depth_1"],
            accumulation=outputs["accumulation"],
        )

        metrics_dict = {}
        images_dict = {
            "img": outputs["rgb"],
            "accumulation": acc,
            "depth": depth,
            "prop_depth_0": prop_depth_0,
            "prop_depth_1": prop_depth_1,
            "normals": outputs["normals"],
        }
        return metrics_dict, images_dict
