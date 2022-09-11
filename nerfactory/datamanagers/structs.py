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
Dataset input structures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torchtyping import TensorType

from nerfactory.cameras.cameras import Cameras
from nerfactory.configs.utils import to_immutable_dict


@dataclass
class PointCloud:
    """Dataclass for a point cloud."""

    xyz: TensorType["num_points", 3] = None
    """the 3D points in point cloud"""
    rgb: TensorType["num_points", 3] = None
    """the color of each point in point cloud"""


@dataclass
class Semantics:
    """Dataclass for semantic labels."""

    stuff_filenames: List[Path]
    """filenames to load "stuff"/background data"""
    stuff_classes: List[str]
    """class labels for "stuff" data"""
    stuff_colors: torch.Tensor
    """color mapping for "stuff" classes"""
    thing_filenames: List[Path]
    """filenames to load "thing"/foreground data"""
    thing_classes: List[str]
    """class labels for "thing" data"""
    thing_colors: torch.Tensor
    """color mapping for "thing" classes"""


@dataclass
class SceneBounds:
    """Data to represent the scene bounds."""

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

    def get_centered_and_scaled_scene_bounds(self, scale_factor: Union[float, torch.Tensor] = 1.0):
        """Returns a new box that has been shifted and rescaled to be centered
        about the origin.

        Args:
            scale_factor: How much to scale the camera origins by.
        """
        return SceneBounds(aabb=(self.aabb - self.get_center()) * scale_factor)

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

    def to_json(self) -> Dict:
        """Returns a json object from the Python object."""
        return {"type": "aabb", "min_point": self.aabb[0].tolist(), "max_point": self.aabb[1].tolist()}

    @staticmethod
    def from_json(json_: Dict) -> "SceneBounds":
        """Returns the an instance of SceneBounds from a json dictionary.

        Args:
            json_: the json dictionary containing scene bound information
        """
        assert json_["type"] == "aabb"
        aabb = torch.tensor([json_[0], json_[1]])
        return SceneBounds(aabb=aabb)

    @staticmethod
    def from_camera_poses(poses: TensorType[..., 3, 4], scale_factor: float) -> "SceneBounds":
        """Returns the instance of SceneBounds that fully envelopes a set of poses

        Args:
            poses: tensor of camera pose matrices
            scale_factor: How much to scale the camera origins by.
        """
        xyzs = poses[..., :3, -1]
        aabb = torch.stack([torch.min(xyzs, dim=0)[0], torch.max(xyzs, dim=0)[0]])
        return SceneBounds(aabb=aabb * scale_factor)


@dataclass
class DatasetInputs:
    """Dataset inputs for the image dataset and the ray generator."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset"""
    alpha_color: Optional[TensorType[3]] = None
    """color of dataset background"""
    scene_bounds: SceneBounds = SceneBounds()
    """scene bounds of dataset"""
    # we support additional input information/formats including mask/depth/semantics.
    additional_inputs: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of additional dataset information (e.g. semantics/point clouds/masks).
    {input_name:
    ... {"func": function to process additional dataset inputs,
    ... "kwargs": dictionary of data to pass into "func"}
    }
    """

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)
