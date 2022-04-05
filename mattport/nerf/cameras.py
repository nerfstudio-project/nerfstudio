"""
Camera Models
"""
from dataclasses import dataclass
from torchtyping import TensorType


@dataclass
class Rays:
    """Camera rays. Can be arbitrary dimension"""

    origin: TensorType[..., 3]
    direction: TensorType[..., 3]


class Camera:
    """Base Camera. Intended to be subclassed"""

    def __init__(self, height, width, camera_to_world) -> None:
        self.height = height
        self.width = width
        self.camera_to_world = camera_to_world

    def get_rays(self, coords: TensorType[..., 2]) -> Rays:
        """Encodes an input tensor.

        Args:
            coords (TensorType[..., 2]): Input tensor to be encoded

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
        """
        raise NotImplementedError

    def get_all_rays(self) -> Rays:
        """Generate rays for entire camera

        Returns:
            Rays: Camera rays of shape [height, width]
        """
        raise NotImplementedError
