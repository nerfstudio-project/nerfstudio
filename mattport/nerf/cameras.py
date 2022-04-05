"""
Camera Models
"""
from dataclasses import dataclass
from torchtyping import TensorType
import torch


@dataclass
class Rays:
    """Camera rays. Can be arbitrary dimension"""

    origin: TensorType[..., 3]
    direction: TensorType[..., 3]


class Camera:
    """Base Camera. Intended to be subclassed"""

    def __init__(self, height: int, width: int, camera_to_world: TensorType[4, 3]) -> None:
        """Initialize camera

        Args:
            height (int): Number of camera sensor rows
            width (int): Number of camera sensor columns
            camera_to_world (TensorType[4, 3]): Pose transorm for camera
        """
        self.height = height
        self.width = width
        self.camera_to_world = camera_to_world

    def generate_rays(self, coords: TensorType[..., 2]) -> Rays:
        """Generates camera rays associated to grid of pixel coordinates.

        Args:
            coords (TensorType[..., 2]): Pixel coordiantes

        Returns:
            Rays: Camera rays associated to pixel coordinates.
        """
        raise NotImplementedError

    def get_image_coords(self) -> TensorType["height", "width", 2]:
        """Get grid of image coordinates

        Returns:
            TensorType["height", "width", 2]: Image coordinates
        """
        coords = torch.meshgrid(torch.arange(self.height), torch.arange(self.width))
        coords = torch.stack(coords, dim=-1)
        return coords

    def generate_all_rays(self) -> Rays:
        """Generate rays for entire camera

        Returns:
            Rays: Camera rays of shape [height, width]
        """
        coords = self.get_image_coords()
        return self.generate_rays(coords)
