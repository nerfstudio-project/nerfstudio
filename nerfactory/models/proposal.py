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
Implementation of Instant NGP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfactory.cameras.rays import RayBundle
from nerfactory.fields.density_field import DensityField
from nerfactory.fields.instant_ngp_field import TCNNInstantNGPField
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.modules.spatial_distortions import SceneContraction
from nerfactory.models.base import Model, ModelConfig
from nerfactory.models.modules.ray_sampler import ProposalNetworkSampler
from nerfactory.models.modules.scene_colliders import NearFarCollider
from nerfactory.optimizers.loss import MSELoss, distortion_loss, interlevel_loss
from nerfactory.renderers.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfactory.utils import colors, visualization


@dataclass
class ProposalModelConfig(ModelConfig):
    """Proposal Model Config"""

    _target: Type = field(default_factory=lambda: ProposalModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 20.0
    """How far along the ray to stop sampling."""
    randomize_background: bool = True
    """Whether to randomize the background color."""
    num_proposal_samples_per_ray: int = 64
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 32
    """Number of samples per ray for the nerf network."""
    num_proposal_network_iterations: int = 2
    """Number of proposal network iterations."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.01
    """Distortion loss multiplier."""


class ProposalModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: ProposalModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Fields
        self.field = TCNNInstantNGPField(self.scene_bounds.aabb, spatial_distortion=SceneContraction())
        self.proposal_network = DensityField(self.scene_bounds.aabb, spatial_distortion=SceneContraction())

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Samplers
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_network_iterations,
        )

        # renderers
        background_color = None if self.config.randomize_background else colors.WHITE
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.proposal_network.parameters()) + list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fn=self.proposal_network.density_fn
        )
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
            outputs["weights_list"],
            outputs["ray_samples_list"],
            self.config.near_plane,
            self.config.far_plane,
        )
        loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
            outputs["weights_list"],
            outputs["ray_samples_list"],
            self.config.near_plane,
            self.config.far_plane,
        )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
