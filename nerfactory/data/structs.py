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
from typing import List, Optional

import torch
from torchtyping import TensorType


@dataclass
class PointCloud:
    """_summary_"""

    xyz: TensorType["num_points", 3] = None
    rgb: TensorType["num_points", 3] = None


@dataclass
class Semantics:
    """_summary_"""

    stuff_classes: List[str] = None
    stuff_colors: List[List[int]] = None
    stuff_filenames: List[str] = None
    thing_classes: List[str] = None
    thing_colors: List[List[int]] = None
    thing_filenames: List[str] = None


@dataclass
class SceneBounds:
    """Data to represent the scene bounds.

    aabb: axis-aligned bounding box
        aabb[0] is the minimum (x,y,z) point. aabb[1] is the maximum (x,y,z) point.
    """

    aabb: TensorType[2, 3] = None

    def get_diagonal_length(self):
        """Returns the longest diagonal length."""
        diff = self.aabb[1] - self.aabb[0]
        length = torch.sqrt((diff**2).sum() + 1e-20)
        return length

    def get_center(self):
        """Returns the center of the box."""
        diff = self.aabb[1] - self.aabb[0]
        return self.aabb[0] + diff / 2.0

    def get_centered_and_scaled_scene_bounds(self, scale_factor=1.0):
        """Returns a new box that has been shifted and rescaled to be centered
        about the origin."""
        return SceneBounds(aabb=(self.aabb - self.get_center()) * scale_factor)


@dataclass
class DatasetInputs:
    """Dataset inputs for the image dataset and the ray generator.

    Args:
        image_filenames: Filenames for the images.
        ...
    """

    image_filenames: List[str]
    downscale_factor: int = 1
    intrinsics: torch.tensor = None
    camera_to_world: torch.tensor = None
    mask_filenames: List[str] = None
    depth_filenames: List[str] = None
    scene_bounds: SceneBounds = SceneBounds()
    semantics: Semantics = Semantics()
    point_cloud: PointCloud = PointCloud()
    alpha_color: Optional[TensorType[3]] = None

    def check_inputs(self):
        """Check the inputs to make sure everything is okay."""
        assert self.intrinsics.dtype == torch.float32
        assert self.camera_to_world.dtype == torch.float32

    def save_to_folder_name(self, data_directory: str):
        """Save the dataset inputs."""
        raise NotImplementedError

    def load_from_folder_name(self, data_directory: str):
        """Load the saved dataset inputs."""
        raise NotImplementedError

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)
