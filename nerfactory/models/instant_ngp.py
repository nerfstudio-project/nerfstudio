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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import nerfacc  # pylint: disable=import-error
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType

from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.fields.instant_ngp_field import field_implementation_to_class
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.models.base import Model
from nerfactory.optimizers.loss import MSELoss
from nerfactory.utils import visualization
from nerfactory.utils.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)


@dataclass
class InstantNGPModelConfig(cfg.ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    field_implementation: Literal["torch", "tcnn"] = "tcnn"  # torch, tcnn, ...
    """one of "torch" or "tcnn", or other fields in 'field_implementation_to_class'"""
    # TODO(ethan): remove the density field specified here
    enable_density_field: bool = False
    """Whether to create a density field to filter samples."""
    num_samples: int = 1024
    """Number of samples in field evaluation. Defaults to 1024,"""
    cone_angle: float = 0.0
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    randomize_background: bool = False
    """Whether to randomize the background color. Defaults to False."""


class NGPModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: InstantNGPModelConfig

    def __init__(self, config: InstantNGPModelConfig, **kwargs) -> None:
        assert config.field_implementation in field_implementation_to_class
        self.field = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # torch or tiny-cuda-nn version
        self.field = field_implementation_to_class[self.config.field_implementation](self.scene_bounds.aabb)

        # to match Ruilong's code naming
        self.scene_aabb = Parameter(self.scene_bounds.aabb.flatten(), requires_grad=False)

        # setup some rendering settings
        render_n_samples = 1024
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
        ).item()

        # setup occupancy field with eval function
        def occ_eval_fn(x: torch.Tensor) -> torch.Tensor:
            """Evaluate occupancy given positions.

            Args:
                x: positions with shape (N, 3).
            Returns:
                occupancy values with shape (N, 1).
            """
            density_after_activation = self.field.density_fn(x)
            # those two are similar when density is small.
            # occupancy = 1.0 - torch.exp(-density_after_activation * render_step_size)
            occupancy = density_after_activation * self.render_step_size
            return occupancy

        # occupancy grid
        self.occ_field = nerfacc.OccupancyField(occ_eval_fn=occ_eval_fn, aabb=self.scene_aabb, resolution=128)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.occ_field.every_n_step,  # will take in step
            )
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None

        def query_fn(positions: TensorType["bs", 3], directions: TensorType["bs", 3], only_density=False):
            if only_density:
                return self.field.density_fn(positions)
            field_outputs = self.field.get_outputs_from_positions_and_direction(positions, directions)
            rgbs = field_outputs[FieldHeadNames.RGB]
            sigmas = field_outputs[FieldHeadNames.DENSITY]
            return rgbs, sigmas

        (
            accumulated_color,
            accumulated_depth,
            accumulated_weight,
            steps_counter,  # pylint: disable=unused-variable
            compact_steps_counter,  # pylint: disable=unused-variable
        ) = nerfacc.volumetric_rendering(
            query_fn=query_fn,
            rays_o=ray_bundle.origins,
            rays_d=ray_bundle.directions,
            scene_aabb=self.scene_aabb,
            scene_occ_binary=self.occ_field.occ_grid_binary,
            scene_resolution=self.occ_field.resolution,
            render_bkgd=torch.ones(3, device=self.device),
            render_step_size=self.render_step_size,
            near_plane=self.config.near_plane,
            stratified=self.training,  # only use stratified sampling during training
        )

        alive_ray_mask = accumulated_weight.squeeze(-1) > 0

        outputs = {
            "rgb": accumulated_color,
            "accumulation": accumulated_weight,
            "depth": accumulated_depth,
            "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict, loss_coefficients):
        image = batch["image"].to(self.device)
        mask = outputs["alive_ray_mask"]
        rgb_loss = self.rgb_loss(image[mask], outputs["rgb"][mask])
        loss_dict = {"rgb_loss": rgb_loss}
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
        alive_ray_mask = visualization.apply_colormap(outputs["alive_ray_mask"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_alive_ray_mask = torch.cat([alive_ray_mask], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "alive_ray_mask": combined_alive_ray_mask,
        }

        return metrics_dict, images_dict
