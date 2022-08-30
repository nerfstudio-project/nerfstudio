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
from typing import Dict, List, Optional, Union

import torch
from torchtyping import TensorType

from nerfactory.cameras.cameras import Cameras
from nerfactory.cameras.rays import RayBundle


@dataclass
class PointCloud:
    """Dataclass for a point cloud."""

    xyz: TensorType["num_points", 3] = None
    rgb: TensorType["num_points", 3] = None


@dataclass
class Semantics:
    """Dataclass for semantic labels."""

    stuff_filenames: List[Path]
    stuff_classes: List[str]
    thing_filenames: List[Path]
    thing_classes: List[str]
    stuff_colors: torch.Tensor
    thing_colors: torch.Tensor


@dataclass
class SceneBounds:
    """Data to represent the scene bounds.

    Attributes:
        aabb: axis-aligned bounding box.
            aabb[0] is the minimum (x,y,z) point.
            aabb[1] is the maximum (x,y,z) point.
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

    def get_centered_and_scaled_scene_bounds(self, scale_factor: Union[float, torch.Tensor] = 1.0):
        """Returns a new box that has been shifted and rescaled to be centered
        about the origin."""
        return SceneBounds(aabb=(self.aabb - self.get_center()) * scale_factor)

    @staticmethod
    def get_normalized_positions(positions: TensorType[..., 3], aabb: TensorType[2, 3]):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box

        Returns:
            positions that are normalized into the range [0, 1].
        """
        aabb_lengths = aabb[1] - aabb[0]
        normalized_positions = (positions - aabb[0]) / aabb_lengths
        return normalized_positions

    def to_json(self) -> Dict:
        """Returns a json object from the Python object."""
        return {"type": "aabb", "min_point": self.aabb[0].tolist(), "max_point": self.aabb[1].tolist()}

    @staticmethod
    def from_json(json_: Dict) -> "SceneBounds":
        """Returns the an instance of SceneBounds from a json dictionary."""
        assert json_["type"] == "aabb"
        aabb = torch.tensor([json_[0], json_[1]])
        return SceneBounds(aabb=aabb)

    @staticmethod
    def from_camera_poses(poses: TensorType[..., 3, 4], scale_factor: float) -> "SceneBounds":
        """Returns the instance of SceneBounds that fully envelopes a set of poses"""
        xyzs = poses[..., :3, -1]
        aabb = torch.stack([torch.min(xyzs, dim=0)[0], torch.max(xyzs, dim=0)[0]])
        return SceneBounds(aabb=aabb * scale_factor)


@dataclass
class DatasetInputs:
    """Dataset inputs for the image dataset and the ray generator.

    Args:
        image_filenames: Filenames for the images.
        intrinsics: Tensor of per-image camera intrisics.
        camera_to_world: Tensor of per-image c2w matrices, in [R | t] format.
        ...
    """

    image_filenames: List[Path]
    cameras: Cameras
    downscale_factor: int = 1
    mask_filenames: Optional[List[Path]] = None
    depth_filenames: Optional[List[Path]] = None
    scene_bounds: SceneBounds = SceneBounds()
    semantics: Optional[Semantics] = None
    point_cloud: PointCloud = PointCloud()
    alpha_color: Optional[TensorType[3]] = None

    def as_dict(self) -> dict:
        """Returns the dataclass as a dictionary."""
        return vars(self)


@dataclass
class DataloaderOutputs:
    """A container for data that is not specific to any dataset. It should be everything
    needed by the renderer and loss calculation. Different datasets and models will probably
    need to subclass from this differently, since different Field modules will require different
    data.

    Args:
        rays (RayBundle): The rays for the image.
        ground_truth_pixels (TensorType["num_pixels", 3]): The ground truth pixels for the image.
    """

    rays: RayBundle  # Raybundle and the cameras will be merged into one thing in a later PR
    cameras: torch.Tensor
    ground_truth_pixels: Optional[TensorType["num_pixels", 3]] = None

    def check_inputs(self, expected_attr: Dict):
        """This will check to make sure that the expected attributes exist.

        This will exist in any subclassed version of this"""
        for attr in expected_attr:
            assert hasattr(self, attr)


@dataclass
class ModelOutputs:
    """A container for outputs from the model.

    This will be passed into the loss function along with the base data container. It should be
    subclassed for when your model outputs different things.

    Args:
        rays (RayBundle): The rays for the image.
        ground_truth_pixels (TensorType["num_pixels", 3]): The ground truth pixels for the image.
    """

    rendered_pixels: Optional[TensorType["num_pixels", 3]] = None
