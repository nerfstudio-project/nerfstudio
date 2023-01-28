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
NeRF implementation that combines many recent advancements, optimized for hi-res images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.model_components.losses import L1Loss, PerceptualLoss, interlevel_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps


@dataclass
class Nerfacto4KModelConfig(NerfactoModelConfig):
    """Nerfacto-4K Model Config"""

    _target: Type = field(default_factory=lambda: Nerfacto4KModel)
    l1_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    perceptual_loss_mult: float = 0.5
    """Distortion loss multiplier."""


class Nerfacto4KModel(NerfactoModel):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: Nerfacto4KModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.l1_loss = L1Loss()
        layer_weights = {"conv1_2": 0, "conv2_2": 0, "conv3_4": 1, "conv4_4": 1, "conv5_4": 1}
        print("we got here.")
        self.perceptual_loss = PerceptualLoss(layer_weights=layer_weights)
        print("we got here 2.")

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        loss_dict["l1_loss"] = self.config.l1_loss_mult * self.l1_loss(image, outputs["rgb"])
        # print(image.shape)
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
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
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
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # normals to RGB for visualization. TODO: use a colormap
        if "normals" in outputs:
            images_dict["normals"] = (outputs["normals"] + 1.0) / 2.0
        if "pred_normals" in outputs:
            images_dict["pred_normals"] = (outputs["pred_normals"] + 1.0) / 2.0

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
