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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, Union

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.configs import base_config as cfg
from nerfstudio.utils import poses as pose_utils


# Camera Optimization related configs
@dataclass
class CameraOptimizerConfig(cfg.InstantiateConfig):
    """Default camera optimizer config. Note: This is a no-op class and will not optimize cameras."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)


@dataclass
class BARFPoseOptimizerConfig(CameraOptimizerConfig):
    """BARF camera optimizer."""

    _target: Type = field(default_factory=lambda: BARFOptimizer)
    noise_variance: Optional[float] = None
    """Optional additional noise to add to poses. Useful for debugging pose optimization."""


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device

    def forward(
        self,
        indices: TensorType["num_cameras"],
    ) -> TensorType["num_cameras", 3, 4]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Tranformation matrices from optimized camera coordinates coordinates
            to given camera coordinates.
        """
        return torch.eye(4, device=self.device).repeat(indices.shape[0], 1, 1)[:, :3, :4]  # no-op (Identity Transform)


class BARFOptimizer(CameraOptimizer):
    """Bundle-Adjusting NeRF (BARF) Pose Optimization"""

    config: BARFPoseOptimizerConfig

    def __init__(self, config: BARFPoseOptimizerConfig, num_cameras: int, device: Union[torch.device, str]) -> None:
        super().__init__(config, num_cameras, device)
        if self.config.noise_variance is not None:
            pose_noise = torch.normal(torch.zeros(self.num_cameras, 6), self.config.noise_variance)
            self.pose_noise = self.exp_map(pose_noise).detach().to(device)
        self.pose_adjustment = nn.Embedding(self.num_cameras, 6, device=device)
        nn.init.zeros_(self.pose_adjustment.weight)

    @classmethod
    def exp_map(cls, tangent_vector: TensorType["num_cameras", 6]) -> TensorType["num_cameras", 3, 4]:
        """Convert SE3 vector into [R|t] transformation matrix.
        Args:
            tangent_vector: SE3 vector
        Returns:
            Respective [R|t] tranformation matrices.
        """

        tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
        tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

        theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
        theta2 = theta**2
        theta3 = theta**3

        near_zero = theta < 1e-2
        non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)
        theta3_nz = torch.where(near_zero, non_zero, theta3)

        # Compute the rotation
        sine = theta.sin()
        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
        one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
        ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
        ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
        ret[:, 0, 1] -= temp[:, 2]
        ret[:, 1, 0] += temp[:, 2]
        ret[:, 0, 2] += temp[:, 1]
        ret[:, 2, 0] -= temp[:, 1]
        ret[:, 1, 2] -= temp[:, 0]
        ret[:, 2, 1] += temp[:, 0]

        # Compute the translation
        sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
        one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
        theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

        ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
        ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
        ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
            tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
        )
        return ret

    def forward(self, indices: TensorType["num_cameras"]) -> TensorType["num_cameras", 3, 4]:
        c_opt2c = self.exp_map(self.pose_adjustment.weight[indices])
        if self.config.noise_variance is not None:
            c_opt2c = pose_utils.multiply(c_opt2c, self.pose_noise[indices])
        return c_opt2c
