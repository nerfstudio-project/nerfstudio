"""
Some ray datastructures.
"""
import random
from dataclasses import dataclass
from typing import Optional
import torch

from torchtyping import TensorType

from mattport.utils.misc import is_not_none


@dataclass
class PointSamples:
    """Samples in space."""

    positions: TensorType[..., 3] = None
    directions: TensorType[..., 3] = None
    camera_indices: TensorType[..., 1] = None


@dataclass
class RayBundle:
    """_summary_

    Returns:
        _type_: _description_
    """

    origins: TensorType["num_rays", 3]
    directions: TensorType["num_rays", 3]
    camera_indices: Optional[TensorType["num_rays"]] = None
    nears: Optional[TensorType["num_rays"]] = None
    fars: Optional[TensorType["num_rays"]] = None
    valid_mask: Optional[TensorType["num_rays"]] = None

    def to_camera_ray_bundle(self, image_height, image_width) -> "CameraRayBundle":
        """Returns a CameraRayBundle from this object."""
        camera_ray_bundle = CameraRayBundle(
            origins=self.origins.view(image_height, image_width, 3),
            directions=self.directions.view(image_height, image_width, 3),
            camera_indices=self.camera_indices.view(image_height, image_width)
            if not isinstance(self.camera_indices, type(None))
            else None,
        )
        return camera_ray_bundle

    def move_to_device(self, device):
        """Move to a device."""
        self.origins = self.origins.to(device)
        self.directions = self.directions.to(device)
        if not isinstance(self.camera_indices, type(None)):
            self.camera_indices = self.camera_indices.to(device)

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

    def get_masked_ray_bundle(self, valid_mask):
        """Return a masked instance of the ray bundle."""
        return RayBundle(
            origins=self.origins[valid_mask],
            directions=self.directions[valid_mask],
            camera_indices=self.camera_indices[valid_mask] if is_not_none(self.camera_indices) else None,
            nears=self.nears[valid_mask] if is_not_none(self.nears) else None,
            fars=self.fars[valid_mask] if is_not_none(self.fars) else None,
            valid_mask=self.valid_mask[valid_mask] if is_not_none(self.valid_mask) else None,
        )


@dataclass
class CameraRayBundle:
    """_summary_"""

    origins: TensorType["image_height", "image_width", 3]
    directions: TensorType["image_height", "image_width", 3]
    camera_indices: Optional[TensorType["image_height", "image_width", 3]] = None
    camera_index: int = None

    def __post_init__(self):
        if not isinstance(self.camera_index, type(None)):
            self.set_camera_indices(self.camera_index)

    def set_camera_indices(self, camera_index: int):
        """Sets the camera indices for a specific camera index."""
        self.camera_index = camera_index
        self.camera_indices = torch.ones_like(self.origins[..., 0]).long() * camera_index

    def get_num_rays(self):
        """Return the number of rays in this bundle."""
        image_height, image_width = self.origins.shape[:2]
        num_rays = image_height * image_width
        return num_rays

    def to_ray_bundle(self) -> RayBundle:
        """_summary_

        Returns:
            RayBundle: _description_
        """
        # TODO(ethan): handle camera_index
        ray_bundle = RayBundle(origins=self.origins.view(-1, 3), directions=self.directions.view(-1, 3))
        return ray_bundle

    def get_row_major_sliced_ray_bundle(self, start_idx, end_idx):
        """Return a RayBundle"""
        camera_indices = (
            self.camera_indices.view(-1)[start_idx:end_idx] if not isinstance(self.camera_indices, type(None)) else None
        )
        return RayBundle(
            origins=self.origins.view(-1, 3)[start_idx:end_idx],
            directions=self.directions.view(-1, 3)[start_idx:end_idx],
            camera_indices=camera_indices,
        )


class RaySamples:
    """_summary_"""

    def __init__(
        self,
        ts: TensorType["num_rays", "num_samples"],
        ray_bundle: RayBundle,
    ) -> None:
        self.ray_bundle = ray_bundle
        self.ts = ts
        self.positions = self.get_positions(ray_bundle)
        self.directions = ray_bundle.directions.unsqueeze(1).repeat(1, self.positions.shape[1], 1)
        self.deltas = self.get_deltas()

    def to_point_samples(self):
        """Convert to PointSamples instance and return."""
        # TODO: make this more interpretable
        return PointSamples(positions=self.positions, directions=self.directions)

    def get_camera_indices(self):
        """Returns camera indices."""
        assert not isinstance(self.ray_bundle.camera_indices, type(None)), "Camera indices cannot be None here."
        camera_indices = self.ray_bundle.camera_indices.unsqueeze(1).repeat(1, self.positions.shape[1])
        return camera_indices

    def get_positions(self, ray_bundle: RayBundle) -> TensorType["num_rays", "num_samples", 3]:
        """Returns positions."""
        return ray_bundle.origins[:, None] + self.ts[:, :, None] * ray_bundle.directions[:, None]

    def get_deltas(self) -> TensorType[..., "num_samples"]:
        """Returns deltas."""
        dists = self.ts[..., 1:] - self.ts[..., :-1]
        dist_inf = 1e10 * torch.ones_like(dists[..., -1:])
        dists = torch.cat([dists, dist_inf], -1)  # [N_rays, N_samples]
        deltas = dists * torch.norm(self.ray_bundle.directions[..., None, :], dim=-1)
        return deltas

    def get_weights(self, densities: TensorType[..., "num_samples", 1]) -> TensorType[..., "num_samples"]:
        """Return weights based on predicted densities

        Args:
            densities (TensorType[..., "num_samples", 1]): Predicted densities for samples along ray

        Returns:
            TensorType[..., "num_samples"]: Weights for each sample
        """

        delta_density = self.deltas * densities[..., 0]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1], dim=-1)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1)).to(densities.device), transmittance], axis=-1
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        return weights
