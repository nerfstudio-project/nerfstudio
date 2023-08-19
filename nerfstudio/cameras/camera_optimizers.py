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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union

import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, OptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    SchedulerConfig,
)
from nerfstudio.utils import poses as pose_utils


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    position_noise_std: float = 0.0
    """Noise to add to initial positions. Useful for debugging."""

    orientation_noise_std: float = 0.0
    """Noise to add to initial orientations. Useful for debugging."""

    optimizer: OptimizerConfig = field(default_factory=lambda: AdamOptimizerConfig(lr=6e-4, eps=1e-15))
    """ADAM parameters for camera optimization."""

    scheduler: SchedulerConfig = field(default_factory=lambda: ExponentialDecaySchedulerConfig(max_steps=10000))
    """Learning rate scheduler for camera optimizer.."""

    param_group: tyro.conf.Suppress[str] = "camera_opt"
    """Name of the parameter group used for pose optimization. Can be any string that doesn't conflict with other
    groups."""


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

        # Initialize pose noise; useful for debugging.
        if config.position_noise_std != 0.0 or config.orientation_noise_std != 0.0:
            assert config.position_noise_std >= 0.0 and config.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [config.position_noise_std] * 3 + [config.orientation_noise_std] * 3, device=device
            )
            self.pose_noise = exp_map_SE3(torch.normal(torch.zeros((num_cameras, 6), device=device), std_vector))
        else:
            self.pose_noise = None

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.device)[:3, :4]

        # Apply initial pose noise.
        if self.pose_noise is not None:
            outputs.append(self.pose_noise[indices, :, :])
        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)
