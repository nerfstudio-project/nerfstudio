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

    xyz = None
    rgb = None


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
