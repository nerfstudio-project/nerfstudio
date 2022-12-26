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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import depth_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class DepthNerfactoModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: DepthNerfactoModel)
    depth_loss_mult: float = 10.
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.05
    """Uncertainty around depth values in meters (defaults to 5cm)."""


class DepthNerfactoModel(NerfactoModel):
    """Depth loss augumented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: DepthNerfactoModelConfig

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            metrics_dict["depth_loss"] = depth_loss(
                weights=outputs["weights_list"][-1],
                ray_samples=outputs["ray_samples_list"][-1],
                termination_depth=batch["depth_image"].to(self.device),
                sigma=torch.tensor([self.config.depth_sigma], device=self.device),
                directions_norm=outputs["directions_norm"],
                is_euclidean=self.config.is_euclidean_depth,
            )

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict is not None and "depth_loss" in metrics_dict
            loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]

        return loss_dict
