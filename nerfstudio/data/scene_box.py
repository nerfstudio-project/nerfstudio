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
Dataset input structures.
"""

from dataclasses import dataclass
from typing import Union, Tuple
import viser.transforms as vtf

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class SceneBox:
    """Data to represent the scene box."""

    aabb: Float[Tensor, "2 3"]
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
    def get_normalized_positions(positions: Float[Tensor, "*batch 3"], aabb: Float[Tensor, "2 3"]):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = aabb[1] - aabb[0]
        normalized_positions = (positions - aabb[0]) / aabb_lengths
        return normalized_positions

    @staticmethod
    def from_camera_poses(poses: Float[Tensor, "*batch 3 4"], scale_factor: float) -> "SceneBox":
        """Returns the instance of SceneBox that fully envelopes a set of poses

        Args:
            poses: tensor of camera pose matrices
            scale_factor: How much to scale the camera origins by.
        """
        xyzs = poses[..., :3, -1]
        aabb = torch.stack([torch.min(xyzs, dim=0)[0], torch.max(xyzs, dim=0)[0]])
        return SceneBox(aabb=aabb * scale_factor)


@dataclass
class OrientedBox:
    R: Float[Tensor, "3 3"]
    """R: rotation matrix."""
    T: Float[Tensor, "3"]
    """T: translation vector."""
    S: Float[Tensor, "3"]
    """S: scale vector."""

    def within(self, pts: Float[Tensor, "n 3"]):
        """Returns a boolean mask indicating whether each point is within the box."""
        R, T, S = self.R, self.T, self.S.to(pts)
        H = torch.eye(4, device=pts.device, dtype=pts.dtype)
        H[:3, :3] = R
        H[:3, 3] = T
        H_world2bbox = torch.inverse(H)
        pts = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
        pts = torch.matmul(H_world2bbox, pts.T).T[..., :3]

        comp_l = torch.tensor(-S / 2)
        comp_m = torch.tensor(S / 2)
        mask = torch.all(torch.concat([pts > comp_l, pts < comp_m], dim=-1), dim=-1)
        return mask

    @staticmethod
    def from_params(
        pos: Tuple[float, float, float], rpy: Tuple[float, float, float], scale: Tuple[float, float, float]
    ):
        """Construct a box from position, rotation, and scale parameters."""
        R = torch.tensor(vtf.SO3.from_rpy_radians(rpy[0], rpy[1], rpy[2]).as_matrix())
        T = torch.tensor(pos)
        S = torch.tensor(scale)
        return OrientedBox(R=R, T=T, S=S)
