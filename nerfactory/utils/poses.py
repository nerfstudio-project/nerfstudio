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
Common 3D pose methods

credit: Alvin Wan (alvinwan@berkeley.edu)
"""

import torch
from torchtyping import TensorType


def to4x4(pose: TensorType[..., 3, 4]) -> TensorType[..., 4, 4]:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate."""
    constants = torch.zeros_like(pose[..., :1, :])
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def inverse(pose: TensorType[..., 3, 4]) -> TensorType[..., 3, 4]:
    """Invert provided pose matrix."""
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)


def multiply(pose_a: TensorType[..., 3, 4], pose_b: TensorType[..., 3, 4]) -> TensorType[..., 3, 4]:
    """Multiply two pose matrices, A @ B."""
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)
