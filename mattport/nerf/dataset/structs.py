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

    xyz: None
    rgb: None


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
    bounding_box: torch.tensor = None
    depth_filenames: List[str] = None
    semantic_filenames: List[str] = None
    point_cloud: PointCloud = None  # TODO(ethan): specify the type here
    alpha_color: Optional[TensorType[3]] = None

    def check_inputs(self):
        """Check the inputs to make sure everything is okay."""
        assert self.intrinsics.dtype == torch.float32
        assert self.camera_to_world.dtype == torch.float32
