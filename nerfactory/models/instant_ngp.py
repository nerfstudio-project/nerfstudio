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
from typing import Dict, List, Literal, Optional, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import nerfactory.cuda as nerfactory_cuda
from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.fields.instant_ngp_field import field_implementation_to_class
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.models.base import Model
from nerfactory.models.modules.ray_sampler import NGPSpacedSampler
from nerfactory.optimizers.loss import MSELoss
from nerfactory.utils import colors, visualization, writer
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
    enable_density_field: bool = True
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

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        assert self.density_field is not None
        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=self.density_field.update_every_num_iters,
                func=self.density_field.update_density_grid,
                kwargs={"density_eval_func": self.field.density_fn},  # type: ignore
            )
        ]

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # torch or tiny-cuda-nn version
        self.field = field_implementation_to_class[self.config.field_implementation](self.scene_bounds.aabb)

        # samplers
        self.sampler = NGPSpacedSampler(num_samples=self.config.num_samples, density_field=self.density_field)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None

        # TODO(ruilongli)
        # - train test difference
        # - visualize "depth_density_grid"
        num_rays = len(ray_bundle)
        device = ray_bundle.origins.device

        ray_samples, packed_info, t_min, t_max = self.sampler(
            ray_bundle, self.field.aabb, cone_angle=self.config.cone_angle, near_plane=self.config.near_plane
        )

        field_outputs = self.field.forward(ray_samples)
        rgbs = field_outputs[FieldHeadNames.RGB]
        sigmas = field_outputs[FieldHeadNames.DENSITY]

        # accumulate all the rays start from zero opacity
        opacities = torch.zeros((num_rays, 1), device=device)
        (
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            alive_ray_mask,
        ) = nerfactory_cuda.VolumeRenderer.apply(
            packed_info,
            ray_samples.frustums.starts,
            ray_samples.frustums.ends,
            sigmas.contiguous(),
            rgbs.contiguous(),
            opacities,
        )
        accumulated_depth = torch.clip(accumulated_depth, t_min[:, None], t_max[:, None])

        if self.config.randomize_background:
            background_colors = torch.rand((num_rays, 3), device=device)  # (num_rays, 3)
        else:
            background_colors = colors.WHITE.to(accumulated_color)  # (3,)

        accumulated_color = accumulated_color + background_colors * (1.0 - accumulated_weight)

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

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
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

        # TODO: return an image dictionary instead of logging here
        writer.put_image(name="Eval Images/img", image=combined_rgb, step=step)
        writer.put_image(name="Eval Images/accumulation", image=combined_acc, step=step)
        writer.put_image(name="Eval Images/depth", image=combined_depth, step=step)
        writer.put_image(name="Eval Images/alive_ray_mask", image=combined_alive_ray_mask, step=step)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()),
            "image_idx": image_idx,
            "ssim": float(ssim),  # type: ignore
            "lpips": float(lpips),
            "acc_min": float(acc.min().item()),
            "acc_max": float(acc.max().item()),
        }
        # TODO(ethan): return an image dictionary
        return metrics_dict
