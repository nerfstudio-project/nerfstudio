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
Implementation of NeRFPlayer (https://arxiv.org/abs/2210.15947) with InstantNGP backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import nerfacc
import torch
from nerfacc import ContractionType
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfplayer_ngp_field import NerfplayerNGPField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.utils import colors


@dataclass
class NerfplayerNGPModelConfig(InstantNGPModelConfig):
    """NeRFPlayer Model Config with InstantNGP backbone.
    Tips for tuning the performance:
    1. If the scene is flickering, this is caused by unwanted high-freq on the temporal dimension.
        Try reducing `temporal_dim` first, but don't be too small, otherwise the dynamic object is blurred.
        Then try increasing the `temporal_tv_weight`. This is the loss for promoting smoothness among the
        temporal channels.
    2. If a faster rendering is preferred, then try reducing `log2_hashmap_size`. If more details are
        wanted, try increasing `log2_hashmap_size`.
    3. If the input cameras are of limited numbers, try reducing `num_levels`. `num_levels` is for
        multi-resolution volume sampling, and has a similar behavior to the freq in NeRF. With a small
        `num_levels`, a blurred rendering will be generated, but it is unlikely to overfit the training views.
    """

    _target: Type = field(default_factory=lambda: NerfplayerNGPModel)
    temporal_dim: int = 64
    """Hashing grid parameter. A higher temporal dim means a higher temporal frequency."""
    num_levels: int = 16
    """Hashing grid parameter."""
    features_per_level: int = 2
    """Hashing grid parameter."""
    log2_hashmap_size: int = 17
    """Hashing grid parameter."""
    base_resolution: int = 16
    """Hashing grid parameter."""
    temporal_tv_weight: float = 1
    """Temporal TV loss balancing weight for feature channels."""
    depth_weight: float = 1e-1
    """depth loss balancing weight for feature channels."""
    train_background_color: Literal["random", "black", "white"] = "random"
    """The training background color that is given to untrained areas."""
    eval_background_color: Literal["random", "black", "white"] = "white"
    """The training background color that is given to untrained areas."""
    disable_viewing_dependent: bool = True
    """Disable viewing dependent effects."""


class NerfplayerNGPModel(NGPModel):
    """NeRFPlayer Model with InstantNGP backbone.

    Args:
        config: NeRFPlayer NGP configuration to instantiate model
    """

    config: NerfplayerNGPModelConfig
    field: NerfplayerNGPField

    def populate_modules(self):
        """Set the fields and modules."""
        Model.populate_modules(self)

        self.field = NerfplayerNGPField(
            aabb=self.scene_box.aabb,
            contraction_type=self.config.contraction_type,
            use_appearance_embedding=self.config.use_appearance_embedding,
            num_images=self.num_train_data,
            temporal_dim=self.config.temporal_dim,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            base_resolution=self.config.base_resolution,
            disable_viewing_dependent=self.config.disable_viewing_dependent,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            contraction_type=self.config.contraction_type,
        )

        # Sampler
        vol_sampler_aabb = self.scene_box.aabb if self.config.contraction_type == ContractionType.AABB else None
        self.sampler = VolumetricSampler(
            scene_aabb=vol_sampler_aabb,
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )  # need to update the density_fn later during forward (for input time)

        # renderers
        self.train_background_color = self.config.train_background_color
        self.eval_background_color = self.config.eval_background_color
        if self.config.train_background_color in ["white", "black"]:
            self.train_background_color = colors.COLORS_DICT[self.config.train_background_color]
        if self.config.eval_background_color in ["white", "black"]:
            self.eval_background_color = colors.COLORS_DICT[self.config.eval_background_color]

        self.renderer_rgb = RGBRenderer()  # will update bgcolor later during forward
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.temporal_distortion = True  # for viewer

    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)

        # update the density_fn of the sampler so that the density is time aware
        self.sampler.density_fn = lambda x: self.field.density_fn(x, ray_bundle.times)
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        # update bgcolor in the renderer; usually random color for training and fixed color for inference
        if self.training:
            self.renderer_rgb.background_color = self.train_background_color
        else:
            self.renderer_rgb.background_color = self.eval_background_color
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        alive_ray_mask = accumulation.squeeze(-1) > 0

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
            "num_samples_per_ray": packed_info[:, 1],
        }
        # adding them to outputs for calculating losses
        if self.training and self.config.depth_weight > 0:
            outputs["ray_indices"] = ray_indices
            outputs["ray_samples"] = ray_samples
            outputs["weights"] = weights
            outputs["sigmas"] = field_outputs[FieldHeadNames.DENSITY]
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        mask = outputs["alive_ray_mask"]
        rgb_loss = self.rgb_loss(image[mask], outputs["rgb"][mask])
        loss_dict = {"rgb_loss": rgb_loss}
        if "depth_image" in batch.keys() and self.config.depth_weight > 0:
            mask = batch["depth_image"] != 0
            # First we calculate the depth value, just like most of the papers.
            loss_dict["depth_loss"] = (outputs["depth"][mask] - batch["depth_image"][mask]).abs().mean()
            # But this is not enough -- it will lead to fog like reconstructions, even with depth supervision.
            # Because this loss only cares about the mean value.
            # (Feel free to try it and see the fog-like effects on DyCheck by commenting out the following loss.)
            # > The fog can be effectively penalized by multiview inputs (i.e., from another viewing point).
            # > However, it is hard to get multiple views under the setting of dynamic scenes.
            # > A new view always indicates another camera, which is expensive and brings sync problems.
            # In DyCheck (https://arxiv.org/abs/2210.13445), surface sparsity regularizer
            # (i.e., `distortion_loss` in nerfstudio) is used to make surface tight.
            # The `distortion_loss` can be used here as well, but personally find it hard to implement...
            # (Due to volume sampling, seems that cuda kernels are needed for efficiently computing the loss.)
            # (Try nerfplayer with nerfacto backbone for distortion loss.)
            # Instead, directly penalizing the empty space is found effective here. Perhaps due to a more
            # "clear" loss (as it is directly applied to the network outputs, rather than post-processed values).
            # But such a loss also has drawbacks: it tends to overfit wrong (or noise) presented in the depth map.
            # Some structures are floating in the air with this loss...
            gt_depth_packed = batch["depth_image"][outputs["ray_indices"]]
            steps = (outputs["ray_samples"].frustums.starts + outputs["ray_samples"].frustums.ends) / 2
            # empty area should not be too close to the given depth, so lets add a margin to the gt depth
            margin = (self.scene_box.aabb.max() - self.scene_box.aabb.min()) / 128
            density_min_mask = (gt_depth_packed - steps > margin) & (gt_depth_packed != 0)
            loss_dict["depth_loss"] += (outputs["sigmas"][density_min_mask[..., 0]].pow(2)).mean() * 1e-2
            loss_dict["depth_loss"] *= self.config.depth_weight
        if self.config.temporal_tv_weight > 0:
            loss_dict["temporal_tv_loss"] = self.config.temporal_tv_weight * self.field.mlp_base.get_temporal_tv_loss()
        return loss_dict
