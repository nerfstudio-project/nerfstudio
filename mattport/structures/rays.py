"""
Some ray datastructures.
"""
from dataclasses import dataclass
from typing import Optional
from torchtyping import TensorType


@dataclass
class CameraRayBundle:
    """_summary_"""

    origins: TensorType["image_height", "image_width", 3]
    directions: TensorType["image_height", "image_width", 3]
    camera_index: int = None


@dataclass
class RayBundle:
    """_summary_

    Returns:
        _type_: _description_
    """

    origins: TensorType["num_rays", 3]
    directions: TensorType["num_rays", 3]
    camera_indices: Optional[TensorType["num_rays"]] = None

    def to_camera_ray_bundle(self, image_height, image_width) -> CameraRayBundle:
        """_summary_

        Args:
            image_height (_type_): _description_
            image_width (_type_): _description_

        Returns:
            CameraRayBundle: _description_
        """
        camera_ray_bundle = CameraRayBundle(
            origins=self.origins.view(image_height, image_width, 3),
            directions=self.directions.view(image_height, image_width, 3),
        )
        return camera_ray_bundle
