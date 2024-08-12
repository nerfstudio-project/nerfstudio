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
from typing import Tuple, Union

import torch
import viser.transforms as vtf
from jaxtyping import Float
from torch import Tensor


@dataclass
class SceneBox:
    """Data to represent the scene box."""

    aabb: Float[Tensor, "2 3"]
    """aabb: axis-aligned bounding box.
    aabb[0] is the minimum (x,y,z) point.
    aabb[1] is the maximum (x,y,z) point."""

    def within(self, pts: Float[Tensor, "n 3"]):
        """Returns a boolean mask indicating whether each point is within the box."""
        return torch.all(pts > self.aabb[0], dim=-1) & torch.all(pts < self.aabb[1], dim=-1)

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


@dataclass
class OrientedSceneBox:
    R: Float[Tensor, "3 3"]
    """R: rotation matrix."""
    T: Float[Tensor, "3"]
    """T: translation vector."""
    S: Float[Tensor, "3"]
    """S: scale vector."""

    def within(self, pts: Float[Tensor, "n 3"]):
        """Returns a boolean mask indicating whether each point is within the box."""
        pts_local = self.to_local_coordinates(pts)
        comp_l = -self.S / 2
        comp_m = self.S / 2
        mask = torch.all(torch.cat([pts_local > comp_l, pts_local < comp_m], dim=-1), dim=-1)
        return mask

    def to_local_coordinates(self, pts: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
        """Transform points to the local coordinate system of the OrientedBox.

        Args:
            pts: Tensor of shape [*batch, 3] where *batch represents any number of leading batch dimensions.

        Returns:
            pts_local: Tensor of shape [*batch, 3] with the points transformed to the local coordinate system.
        """
        R, T, _ = self.R, self.T, self.S.to(pts)

        # Construct the homogeneous transformation matrix H_world2bbox
        H = torch.eye(4, device=pts.device, dtype=pts.dtype)
        H[:3, :3] = R
        H[:3, 3] = T
        H_world2bbox = torch.inverse(H)  # [4, 4]

        # Add homogeneous coordinate to pts to make it [*batch, 4]
        ones = torch.ones(*pts.shape[:-1], 1, device=pts.device, dtype=pts.dtype)  # shape: [*batch, 1]
        pts_homogeneous = torch.cat((pts, ones), dim=-1)  # shape: [*batch, 4]

        # Reshape pts_homogeneous to [-1, 4] for batched matrix multiplication
        original_shape = pts_homogeneous.shape
        pts_homogeneous_flat = pts_homogeneous.view(
            -1, 4
        )  # Flatten to 2D tensor for matmul: [N, 4] where N is the total number of points

        # Perform batched matrix multiplication
        pts_local_flat = torch.matmul(H_world2bbox, pts_homogeneous_flat.T).T[..., :3]  # [N, 3]

        # Reshape pts_local back to the original shape minus the homogeneous coordinate
        pts_local = pts_local_flat.view(*original_shape[:-1], 3)  # Reshape to [*batch, 3]

        return pts_local

    def normalize_positions(self, pts: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
        """Returns normalized positions inside the OrientedBox.

        Args:
            pts: The xyz positions to be normalized.

        Returns:
            Normalized positions within the OBB in range [0, 1].
        """
        pts_local = self.to_local_coordinates(pts)
        normalized_positions = (pts_local + (self.S / 2)) / self.S
        return normalized_positions

    @staticmethod
    def from_params(
        pos: Tuple[float, float, float], rpy: Tuple[float, float, float], scale: Tuple[float, float, float]
    ):
        """Construct a box from position, rotation, and scale parameters."""
        R = torch.tensor(vtf.SO3.from_rpy_radians(rpy[0], rpy[1], rpy[2]).as_matrix())
        T = torch.tensor(pos)
        S = torch.tensor(scale)
        return OrientedBox(R=R, T=T, S=S)
