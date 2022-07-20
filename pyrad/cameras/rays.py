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
Some ray datastructures.
"""
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torchtyping import TensorType

from pyrad.utils.math import Gaussians, conical_frustum_to_gaussian
from pyrad.utils.misc import is_not_none
from pyrad.utils.tensor_dataclass import TensorDataclass


@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum.

    Args:
        origins (TensorType[..., 3]): xyz coordinate for ray origin.
        directions (TensorType[..., 3]): Direction of ray.
        starts (TensorType[..., num_samples, 1]): Where the frustum starts along a ray.
        ends (TensorType[..., num_samples, 1]): Where the frustum ends along a ray.
        pixel_area (TensorType[..., 1]): Projected area of pixel a distance 1 away from origin.
    """

    origins: TensorType[..., 3]
    directions: TensorType[..., 3]
    starts: TensorType[..., 1]
    ends: TensorType[..., 1]
    pixel_area: TensorType[..., 1]

    def get_positions(self) -> TensorType[..., 3]:
        """Calulates "center" position of frustum. Not weighted by mass.

        Returns:
            TensorType[..., 3]: xyz positions.
        """
        return self.origins + self.directions * (self.starts + self.ends) / 2

    def get_gaussian_blob(self) -> Gaussians:
        """Calculates of guassian approximation of conical frustum.

        Resturns:
            Gaussians: Conical frustums approximated by gaussian distribution.
        """
        # Cone radius is set such that the square pixel_area matches the cone area.
        cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            radius=cone_radius,
        )

    @classmethod
    def get_mock_frustum(cls, device="cpu") -> "Frustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            Frustums: A size 1 frustum with meaningless values.
        """
        return Frustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
        )


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray

    Args:
        frustums (Frustums): Frustums along ray.
        camera_indices (TensorType[..., 1]): Camera index.
        valid_mask (TensorType[..., 1]): Rays that are valid.
        deltas (TensorType[..., 1]): "width" of each sample.
    """

    frustums: Frustums
    camera_indices: TensorType[..., 1] = None
    valid_mask: TensorType[..., 1] = None
    deltas: TensorType[..., 1] = None

    def get_weights(self, densities: TensorType[..., "num_samples", 1]) -> TensorType[..., "num_samples", 1]:
        """Return weights based on predicted densities

        Args:
            densities (TensorType[..., "num_samples", 1]): Predicted densities for samples along ray

        Returns:
            TensorType[..., "num_samples", 1]: Weights for each sample
        """

        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        # mip-nerf version of transmittance calculation:
        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1)).to(densities.device), transmittance], dim=-2
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

    def apply_masks(self) -> "RaySamples":
        """Use valid_mask to mask samples.

        Returns:
            RaySamples: New set of masked samples.
        """
        if is_not_none(self.valid_mask):
            return self[self.valid_mask[..., 0]]
        return self


@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters.

    Args:
        origins (TensorType[..., 3]): Ray origins (XYZ)
        directions (TensorType[..., 3]): Unit ray direction vector
        pixel_area (TensorType[..., 1]): Projected area of pixel a distance 1 away from origin.
        camera_indices (TensorType[..., 1]): Camera indices
        nears (TensorType[..., 1]): Distance along ray to start sampling
        fars (TensorType[..., 1]): Rays Distance along ray to stop sampling
        valid_mask (TensorType[..., 1]): Rays that are valid
    """

    origins: TensorType["num_rays", 3]
    directions: TensorType["num_rays", 3]
    pixel_area: TensorType["num_rays", 1]
    camera_indices: Optional[TensorType["num_rays", 1]] = None
    nears: Optional[TensorType["num_rays", 1]] = None
    fars: Optional[TensorType["num_rays", 1]] = None
    valid_mask: Optional[TensorType["num_rays", 1]] = None
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
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

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
        return self[indices]

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indicies.

        Args:
            start_idx (int): Start index of RayBundle chunk.
            end_idx (int): End index of RayBundle chunk.

        Returns:
            RayBundle: Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]

    def get_ray_samples(
        self, bin_starts: TensorType[..., "num_samples", 1], bin_ends: TensorType[..., "num_samples", 1]
    ) -> RaySamples:
        """Produces samples for each ray by projection points along the ray direction.


        Args:
            bin_starts (TensorType[..., "num_samples", 1]): Distance from origin to start of bin.
            bin_ends (TensorType[..., "num_samples", 1]): Distance from origin to end of bin.

        Returns:
            RaySamples: Samples projected along ray.
        """
        device = self.origins.device

        valid_mask = torch.ones((bin_starts.shape), dtype=torch.bool, device=device)

        dists = bin_ends - bin_starts  # [..., N_samples, 1]
        deltas = dists * torch.norm(self.directions[:, :], dim=-1)[..., None, None]

        if is_not_none(self.camera_indices):
            camera_indices = self.camera_indices[..., None]
        else:
            camera_indices = None

        frustums = Frustums(
            origins=self.origins[..., None, :],  # [..., 1, 3]
            directions=self.directions[..., None, :],  # [..., 1, 3]
            starts=bin_starts,  # [..., N_samples, 1]
            ends=bin_ends,  # [..., N_samples, 1]
            pixel_area=self.pixel_area[..., None, :],  # [..., 1, 1]
        ).to(device)

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=camera_indices,  # [..., 1, 1]
            valid_mask=valid_mask,  # [..., N_samples, 1]
            deltas=deltas,  # [..., N_samples, 1]
        )

        return ray_samples
