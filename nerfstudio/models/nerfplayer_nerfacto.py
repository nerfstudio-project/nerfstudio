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
NeRFPlayer (https://arxiv.org/abs/2210.15947) implementation with nerfacto backbone.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Type

import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfplayer_nerfacto_field import (
    NerfplayerNerfactoField,
    TemporalHashMLPDensityField,
)
from nerfstudio.model_components.losses import (
    MSELoss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class NerfplayerNerfactoModelConfig(NerfactoModelConfig):
    """Nerfplayer Model Config with Nerfacto backbone"""

    _target: Type = field(default_factory=lambda: NerfplayerNerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample"] = "random"
    """Whether to randomize the background color. (Random is reported to be better on DyCheck.)"""
    num_levels: int = 16
    """Hashing grid parameter."""
    features_per_level: int = 2
    """Hashing grid parameter."""
    log2_hashmap_size: int = 18
    """Hashing grid parameter."""
    temporal_dim: int = 32
    """Hashing grid parameter. A higher temporal dim means a higher temporal frequency."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "temporal_dim": 32, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "temporal_dim": 32, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    distortion_loss_mult: float = 1e-2
    """Distortion loss multiplier."""
    temporal_tv_weight: float = 1
    """Temporal TV balancing weight for feature channels."""
    depth_weight: float = 1e-1
    """depth loss balancing weight for feature channels."""


class NerfplayerNerfactoModel(NerfactoModel):
    """Nerfplayer model with Nerfacto backbone.

    Args:
        config: Nerfplayer configuration to instantiate model
    """

    config: NerfplayerNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        Model.populate_modules(self)

        scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfplayerNerfactoField(
            self.scene_box.aabb,
            temporal_dim=self.config.temporal_dim,
            num_levels=self.config.num_levels,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = TemporalHashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = TemporalHashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
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
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")  # for depth loss
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.temporal_distortion = True  # for viewer

    def get_outputs(self, ray_bundle: RayBundle):
        assert ray_bundle.times is not None, "Time not provided."
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=[functools.partial(f, times=ray_bundle.times) for f in self.density_fns]
        )
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            outputs["normals"] = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            outputs["pred_normals"] = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if "depth_image" in batch.keys() and self.config.depth_weight > 0:
                mask = (batch["depth_image"] != 0).view([-1])
                loss_dict["depth_loss"] = 0
                l = lambda x: self.config.depth_weight * (x - batch["depth_image"][mask]).pow(2).mean()
                loss_dict["depth_loss"] = l(outputs["depth"][mask])
                for i in range(self.config.num_proposal_iterations):
                    loss_dict["depth_loss"] += l(outputs[f"prop_depth_{i}"][mask])
            if self.config.temporal_tv_weight > 0:
                loss_dict["temporal_tv_loss"] = self.field.mlp_base.get_temporal_tv_loss()
                for net in self.proposal_networks:
                    loss_dict["temporal_tv_loss"] += net.encoding.get_temporal_tv_loss()
                loss_dict["temporal_tv_loss"] *= self.config.temporal_tv_weight
        return loss_dict
