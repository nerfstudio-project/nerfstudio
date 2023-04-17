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
Dataset input structures.
"""

from dataclasses import dataclass
from typing import Union

import torch
from torchtyping import TensorType


@dataclass
class SceneBox:
    """Data to represent the scene box."""

    aabb: TensorType[2, 3] = None
    """aabb: axis-aligned bounding box.
    aabb[0] is the minimum (x,y,z) point.
    aabb[1] is the maximum (x,y,z) point."""

    def get_diagonal_length(self):
        """Returns the longest diagonal length."""
        diff = self.aabb[1] - self.aabb[0]
        length = torch.sqrt((diff**2).sum() + 1e-20)
        return length

    def get_center(self):
        """Returns the center of the box."""
        diff = self.aabb[1] - self.aabb[0]
        return self.aabb[0] + diff / 2.0

    def get_centered_and_scaled_scene_box(self, scale_factor: Union[float, torch.Tensor] = 1.0):
        """Returns a new box that has been shifted and rescaled to be centered
        about the origin.

        Args:
            scale_factor: How much to scale the camera origins by.
        """
        return SceneBox(aabb=(self.aabb - self.get_center()) * scale_factor)

    @staticmethod
    def get_normalized_positions(positions: TensorType[..., 3], aabb: TensorType[2, 3]):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = aabb[1] - aabb[0]
        normalized_positions = (positions - aabb[0]) / aabb_lengths
        return normalized_positions

    @staticmethod
    def from_camera_poses(poses: TensorType[..., 3, 4], scale_factor: float) -> "SceneBox":
        """Returns the instance of SceneBox that fully envelopes a set of poses

        Args:
            poses: tensor of camera pose matrices
            scale_factor: How much to scale the camera origins by.
        """
        xyzs = poses[..., :3, -1]
        aabb = torch.stack([torch.min(xyzs, dim=0)[0], torch.max(xyzs, dim=0)[0]])
        return SceneBox(aabb=aabb * scale_factor)
