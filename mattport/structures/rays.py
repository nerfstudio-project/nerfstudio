"""
Some ray datastructures.
"""
import random
from dataclasses import dataclass
from typing import Optional

from torchtyping import TensorType


@dataclass
class RayBundle:
    """_summary_

    Returns:
        _type_: _description_
    """

    origins: TensorType["num_rays", 3]
    directions: TensorType["num_rays", 3]
    camera_indices: Optional[TensorType["num_rays"]] = None

    def to_camera_ray_bundle(self, image_height, image_width) -> "CameraRayBundle":
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

    def __len__(self):
        num_rays = self.origins.shape[0]
        return num_rays

    def sample(self, num_rays: int):
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays (int):

        Returns:
            RayBundle: _description_
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return RayBundle(
            origins=self.origins[indices],
            directions=self.directions[indices],
            camera_indices=self.camera_indices[indices],
        )


@dataclass
class CameraRayBundle:
    """_summary_"""

    origins: TensorType["image_height", "image_width", 3]
    directions: TensorType["image_height", "image_width", 3]
    camera_index: int = None


class RaySamples:
    """_summary_
    """
    def __init__(
        self,
        t_min: TensorType["num_rays"],
        t_max: TensorType["num_rays"],
        ts: TensorType["num_rays", "num_samples+1"],
        ray_bundle: RayBundle,
    ) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.ts = ts
        self.positions = self.get_positions(ray_bundle)
        self.directions = ray_bundle.directions.unsqueeze(1).repeat(1, self.positions.shape[1], 1)
        self.deltas = self.get_deltas()

    def get_positions(self, ray_bundle: RayBundle) -> TensorType["num_rays", "num_samples", 3]:
        """Returns positions."""
        # TODO(ethan): check with Matt about this
        t_mid = (self.ts[:, 1:] + self.ts[:, :-1]) / 2  # (num_rays, num_samples)
        return ray_bundle.origins[:, None] + t_mid[:, :, None] * ray_bundle.directions[:, None]

    def get_deltas(self) -> TensorType["num_samples"]:
        """Returns deltas."""
        deltas = self.ts[..., 1:] - self.ts[..., :-1]
        # TODO(ethan): check this with Matt
        return deltas
