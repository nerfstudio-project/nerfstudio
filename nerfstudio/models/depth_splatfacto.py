# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
A minimal implementation of depth splatfacto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, mse_depth_loss, depth_ranking_loss


from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    depth_loss_mult: float = 0.0001
    """Lambda of the depth loss."""
    depth_loss_type: DepthLossType = DepthLossType.MSE
    """Depth loss type."""
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""

class DepthSplatfactoModel(SplatfactoModel):
    """Nerfstudio's implementation of Depth Supervised Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: DepthSplatfactoModelConfig

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            if self.config.depth_loss_type in (DepthLossType.MSE,):
                metrics_dict["depth_loss"] = torch.Tensor([0.0]).to(self.device)
                termination_depth = batch["depth_image"].to(self.device)

                output_depth_shape = outputs["depth"].shape[:2]
                termination_depth = F.interpolate(termination_depth.permute(2, 0, 1).unsqueeze(0), size=(output_depth_shape[0], output_depth_shape[1]), mode='bilinear', align_corners=False)
                # Remove the extra dimensions added by unsqueeze and permute
                termination_depth = termination_depth.squeeze(0).permute(1, 2, 0)

                metrics_dict["depth_loss"] = mse_depth_loss(
                    termination_depth, outputs["depth"])
                
            elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["depth"], batch["depth_image"].to(self.device)
                )
            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")
        return metrics_dict

   
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict
        
       