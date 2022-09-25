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

from calendar import c
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfactory.cameras.rays import RayBundle
from nerfactory.datamanagers.structs import SceneBounds
from nerfactory.fields.instant_ngp_field import TCNNInstantNGPField
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.proposal_field import DensityField
from nerfactory.models.base import Model, ModelConfig
from nerfactory.models.modules.ray_sampler import (
    PDFSampler,
    ProposalNetworkSampler,
    UniformSampler,
    VolumetricSampler,
)
from nerfactory.models.modules.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfactory.optimizers.loss import MSELoss, interlevel_loss, distortion_loss
from nerfactory.renderers.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfactory.utils import colors, visualization
from nerfactory.utils.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfactory.fields.modules.spatial_distortions import SceneContraction


@dataclass
class ProposalModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: ProposalModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    max_num_samples_per_ray: int = 1024
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    cone_angle: float = 0.0
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    randomize_background: bool = False
    """Whether to randomize the background color."""
    num_proposal_samples_per_ray: int = 64
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 32
    """Number of samples per ray for the nerf network."""
    num_proposal_network_iterations: int = 1
    """Number of proposal network iterations."""
    interlevel_loss_mult: float = 1.0
    distortion_loss_mult: float = 0.01


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
        self.field = TCNNInstantNGPField(self.scene_bounds.aabb)
        self.proposal_network = DensityField(self.scene_bounds.aabb)

        # Collider
        self.collider = AABBBoxCollider(scene_bounds=self.scene_bounds, near_plane=self.config.near_plane)

        # Samplers
        self.pdf_sampler = PDFSampler(include_original=False)
        self.uniform_sampler = UniformSampler()

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

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return []

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # TODO: maybe use different learning rates
        param_groups["fields"] = list(self.field.parameters()) + list(self.proposal_network.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        n = self.config.num_proposal_network_iterations
        weights_list = []
        sdist_list = []
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.config.num_proposal_samples_per_ray if is_prop else self.config.num_nerf_samples_per_ray
            if i_level == 0:
                # need to start with some samples
                ray_samples = self.uniform_sampler(ray_bundle, num_samples=num_samples)
            else:
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, weights, num_samples)
            mlp = self.proposal_network if is_prop else self.field
            field_outputs = mlp(ray_samples)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights_list.append(weights[..., 0])  # (num_rays, num_samples)

            # normalize the intervals
            nears = ray_bundle.nears[..., None]
            fars = ray_bundle.fars[..., None]
            nears_m_fars = fars - nears
            starts = (ray_samples.frustums.starts - nears) / nears_m_fars
            ends = (ray_samples.frustums.ends - nears) / nears_m_fars

            sdist = torch.cat([starts[..., 0], ends[..., -1:, 0]], dim=-1)  # (num_rays, num_samples + 1)
            sdist_list.append(sdist.detach())  # NOTE: detaching here

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        outputs["weights_list"] = weights_list
        outputs["sdist_list"] = sdist_list
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
            outputs["weights_list"], outputs["sdist_list"]
        )
        loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
            outputs["weights_list"], outputs["sdist_list"]
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
