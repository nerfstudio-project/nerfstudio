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
Pose and Intrinsics Optimizers
"""

import torch
from torch import nn
from torchtyping import TensorType


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    def forward(
        self,
        indices: TensorType["num_cameras"],
    ) -> TensorType["num_cameras", 3, 4]:
        return torch.eye(indices.shape[0], 4)[..., :3, :4]  # no-op


class BARFOptimizer(CameraOptimizer):
    def __init__(self, num_cameras: int, noise_variance: float = 0.01) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        pose_noise = torch.normal(torch.zeros(self.num_cameras, 6), noise_variance)
        self.pose_noise = nn.Parameter(self.exp_map(pose_noise), requires_grad=False)
        self.pose_adjustment = nn.Embedding(self.num_cameras, 6)
        nn.init.zeros_(self.pose_adjustment.weight)

    @classmethod
    def exp_map(cls, tangent_vector: TensorType["num_cameras", 6]) -> TensorType["num_cameras", ...]:
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
        sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, theta.sin() / theta_nz)
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

    @classmethod
    def compose(cls, se3_1: torch.Tensor, se3_2: torch.Tensor):  # type: ignore
        batch_size = max(se3_1.shape[0], se3_2.shape[0])
        se3_2 = se3_2.to(se3_1.device)
        ret = torch.zeros(batch_size, 3, 4, dtype=se3_1.dtype, device=se3_1.device)
        ret[:, :, :3] = se3_1[:, :, :3] @ se3_2[:, :, :3]
        ret[:, :, 3] = se3_1[:, :, 3]
        ret[:, :, 3:] += se3_1[:, :, :3] @ se3_2[:, :, 3:]
        return ret

    def forward(self, indices: Optional[TensorType["num_cameras"]] = None) -> TensorType["num_cameras", 3, 4]:

        pose_noise = self.pose_noise[indices]
        pose_adjustment = self.exp_map(self.pose_adjustment.weight[indices])
        c2c_prime = self.compose(pose_noise, pose_adjustment)
        return c2c_prime
