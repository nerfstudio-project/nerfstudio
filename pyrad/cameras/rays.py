"""
Some ray datastructures.
"""
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torchtyping import TensorType

from pyrad.utils.misc import is_not_none


@dataclass
class Frustums:
    """Describes region of space as a frustum.

    Args:
        origins (TensorType[..., 3]): xyz coordinate for ray origin.
        directions (TensorType[..., 3]): Direction of ray.
        frustum_starts (TensorType[..., 1]): Where the frustum starts along a ray.
        frustum_ends (TensorType[..., 3]): Where the frustim ends along a ray.
        pixel_area (TensorType[..., 1]): Projected area of pixel a distance 1 away from origin.
    """

    origins: TensorType[..., 3]
    directions: TensorType[..., 3]
    frustum_starts: TensorType[..., 1]
    frustum_ends: TensorType[..., 1]
    pixel_area: TensorType[..., 1]

    def get_positions(self) -> TensorType[..., 3]:
        """Returns "center" position of frustum. Not weighted by mass."""
        return self.origins + self.directions * (self.frustum_starts + self.frustum_ends) / 2

    @classmethod
    def get_mock_frustum(cls) -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            Frustums: A size 1 frustum with meaningless values.
        """
        device = cls.origins.device
        return Frustums(
            origins=torch.ones((1, 3), device=device),
            directions=torch.ones((1, 3), device=device),
            frustum_starts=torch.ones((1, 1), device=device),
            frustum_ends=torch.ones((1, 1), device=device) + 1,
            pixel_area=torch.ones((1, 1), device=device),
        )

    def apply_masks(self, mask: TensorType) -> "Frustums":
        """Use valid_mask to mask samples.

        Args:
            mask (TensorType): Frustums to keep

        Returns:
            Frustum: New set of masked frustums.
        """
        return Frustums(
            origins=self.origins[mask],
            directions=self.directions[mask],
            frustum_starts=self.frustum_starts[mask],
            frustum_ends=self.frustum_ends[mask],
            pixel_area=self.pixel_area[mask],
        )


@dataclass
class PointSamples:
    """Samples in space.

    Args:
        frustims (Frustums): Frustums along ray.
        directions (TensorType[..., 3]): Unit direction vector.
        camera_indices (TensorType[..., 1]): Camera index.
        valid_mask (TensorType[...]): Rays that are valid.
    """

    frustums: Frustums
    camera_indices: TensorType[..., 1] = None
    valid_mask: TensorType[...] = None

    def apply_masks(self) -> "PointSamples":
        """Use valid_mask to mask samples.

        Returns:
            PointSamples: New set of masked samples.
        """
        if is_not_none(self.valid_mask):
            frustums = self.frustums.apply_masks(self.valid_mask)
            camera_indices = (
                self.camera_indices[self.valid_mask] if is_not_none(self.camera_indices) else self.camera_indices
            )
            return PointSamples(
                frustums=frustums,
                camera_indices=camera_indices,
            )
        return PointSamples(
            frustums=self.frustums,
            camera_indices=self.camera_indices,
        )


@dataclass
class RaySamples:
    """Samples along a ray

    Args:
        frustums (Frustums): Frustums along ray.
        camera_indices (TensorType[..., 1]): Camera index.
        valid_mask (TensorType[...]): Rays that are valid.
        bins (TensorType[..., 1]): frustum bins along ray.
        deltas )TensorType[..., 1]): "width" of each sample.
    """

    frustums: TensorType[..., 3]
    camera_indices: TensorType[..., 1] = None
    valid_mask: TensorType[...] = None
    bins: TensorType[..., 1] = None
    deltas: TensorType[..., 1] = None

    def to_point_samples(self) -> PointSamples:
        """Convert to PointSamples instance and return."""
        # TODO: make this more interpretable
        return PointSamples(
            frustums=self.frustums,
            camera_indices=self.camera_indices,
            valid_mask=self.valid_mask,
        )

    def get_weights(self, densities: TensorType[..., "num_samples", 1]) -> TensorType[..., "num_samples"]:
        """Return weights based on predicted densities

        Args:
            densities (TensorType[..., "num_samples", 1]): Predicted densities for samples along ray

        Returns:
            TensorType[..., "num_samples"]: Weights for each sample
        """

        delta_density = self.deltas * densities[..., 0]
        alphas = 1 - torch.exp(-delta_density)

        # mip-nerf version of transmittance calculation:
        transmittance = torch.cumsum(delta_density[..., :-1], dim=-1)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1)).to(densities.device), transmittance], axis=-1
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        # most nerf codebases do the following:
        # transmittance = torch.cat(
        #     [torch.ones((*alphas.shape[:1], 1)).to(densities.device), 1.0 - alphas + 1e-10], dim=-1
        # )
        # transmittance = torch.cumprod(transmittance, dim=-1)[..., :-1]  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        return weights

    def set_valid_mask(self, valid_mask: TensorType[..., "num_samples"]) -> None:
        """Sets valid mask"""
        self.valid_mask = valid_mask


@dataclass
class RayBundle:
    """A bundle of ray parameters.

    Args:
        origins (TensorType[..., 3]): Ray origins (XYZ)
        directions (TensorType[..., 3]): Unit ray direction vector
        pixel_area (TensorType[..., 1]): Projected area of pixel a distance 1 away from origin.
        camera_indices (TensorType[..., 1]): Camera indices
        nears (TensorType[..., 1]): Distance along ray to start sampling
        fars (TensorType[..., 1]): Rays Distance along ray to stop sampling
        valid_mask (TensorType[...]): Rays that are valid
    """

    origins: TensorType["num_rays", 3]
    directions: TensorType["num_rays", 3]
    pixel_area: TensorType["num_rays", 1]
    camera_indices: Optional[TensorType["num_rays", 1]] = None
    nears: Optional[TensorType["num_rays"]] = None
    fars: Optional[TensorType["num_rays"]] = None
    valid_mask: Optional[TensorType["num_rays"]] = None
    num_rays_per_chunk: int = None

    def move_to_device(self, device: torch.device) -> None:
        """Move bundle data to a device.

        Args:
            device (torch.device): Device to move tensors to.
        """
        self.origins = self.origins.to(device)
        self.directions = self.directions.to(device)
        if is_not_none(self.camera_indices):
            self.camera_indices = self.camera_indices.to(device)

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all of the the camera indices to a specific camera index.

        Args:
            camera_index (int): Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0]).long() * camera_index

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def sample(self, num_rays: int) -> "RayBundle":
        """Returns a RayBundle as a subset of rays.

        Args:
            num_rays (int): Number of rays in output RayBundle

        Returns:
            RayBundle: RayBundle with subset of rays.
        """
        assert num_rays <= len(self)
        indices = random.sample(range(len(self)), k=num_rays)
        return RayBundle(
            origins=self.origins[indices],
            directions=self.directions[indices],
            pixel_area=self.pixel_area[indices],
            camera_indices=self.camera_indices[indices],
        )

    def get_masked_ray_bundle(self, valid_mask: TensorType) -> "RayBundle":
        """Return a masked instance of the ray bundle.

        Args:
            valid_mask (TensorType): Mask of rays to keep.

        Returns:
            RayBundle: RayBundle of masked rays.
        """
        return RayBundle(
            origins=self.origins[valid_mask],
            directions=self.directions[valid_mask],
            pixel_area=self.pixel_area[valid_mask],
            camera_indices=self.camera_indices[valid_mask] if is_not_none(self.camera_indices) else None,
            nears=self.nears[valid_mask] if is_not_none(self.nears) else None,
            fars=self.fars[valid_mask] if is_not_none(self.fars) else None,
            valid_mask=self.valid_mask[valid_mask] if is_not_none(self.valid_mask) else None,
        )

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indicies.

        Args:
            start_idx (int): Start index of RayBundle chunk.
            end_idx (int): End index of RayBundle chunk.

        Returns:
            RayBundle: Flattened RayBundle with end_idx-start_idx rays.

        """
        camera_indices = (
            self.camera_indices.view(-1)[start_idx:end_idx] if not isinstance(self.camera_indices, type(None)) else None
        )
        return RayBundle(
            origins=self.origins.view(-1, 3)[start_idx:end_idx],
            directions=self.directions.view(-1, 3)[start_idx:end_idx],
            pixel_area=self.pixel_area.view(-1, 1)[start_idx:end_idx],
            camera_indices=camera_indices,
        )

    def get_ray_samples(self, bins: TensorType["num_rays", "num_samples+1"]) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction.


        Args:
            bins (TensorType["num_rays", "num_samples+1"]): Distance from origin of sample points.

        Returns:
            RaySamples: Samples projected along ray.
        """
        device = self.origins.device
        num_samples = bins.shape[-1] - 1

        valid_mask = torch.ones((bins.shape[0], num_samples), dtype=torch.bool, device=device)

        dists = bins[:, 1:] - bins[:, :-1]  # [N_rays, N_samples]
        deltas = dists * torch.norm(self.directions[:, None, :], dim=-1)

        if is_not_none(self.camera_indices):
            camera_indices = self.camera_indices.unsqueeze(1).repeat(1, num_samples)
        else:
            camera_indices = None

        frustums = Frustums(
            origins=self.origins.unsqueeze(1).repeat(1, num_samples, 1),
            directions=self.directions.unsqueeze(1).repeat(1, num_samples, 1),
            frustum_starts=bins[:, :-1, None].to(device),
            frustum_ends=bins[:, 1:, None].to(device),
            pixel_area=self.pixel_area.unsqueeze(1).repeat(1, num_samples, 1).to(device),
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,
            valid_mask=valid_mask,
            bins=bins,
            deltas=deltas,
        )

        return ray_samples
