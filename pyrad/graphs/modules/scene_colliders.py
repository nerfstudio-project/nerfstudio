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
Ray generator.
"""
import torch
from torch import nn
from torchtyping import TensorType
from pyrad.data.structs import SceneBounds
from pyrad.cameras.rays import RayBundle


class SceneBoundsCollider(nn.Module):
    """Module for setting near and far values for rays."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        super().__init__()

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        """To be implemented."""
        raise NotImplementedError


class AABBBoxCollider(SceneBoundsCollider):
    """Module for colliding rays with the scene bounds to compute near and far values."""

    def __init__(self, scene_bounds: SceneBounds, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scene_bounds = scene_bounds

    @classmethod
    def intersect_with_aabb(
        cls, rays_o: TensorType["num_rays", 3], rays_d: TensorType["num_rays", 3], aabb: TensorType[2, 3]
    ):
        """_summary_

        Args:
            rays_o (torch.tensor): (num_rays, 3)
            rays_d (torch.tensor): (num_rays, 3)
            aabb (torch.tensor): (2, 3) This is [min point (x,y,z), max point (x,y,z)]
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

        nears = torch.max(
            torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
        ).values
        fars = torch.min(
            torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
        ).values

        # fars < 0: means the ray is behind the camera
        # nears > fars: means no intersection
        valid_mask = ((fars > nears).float() * (fars > 0).float()).bool()
        nears[nears < 0.0] = 0.0
        nears[~valid_mask] = 0.0
        fars[~valid_mask] = 0.0
        return nears, fars, valid_mask

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        """Intersects the rays with the scene bounds and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.
        """
        aabb = self.scene_bounds.aabb
        nears, fars, valid_mask = self.intersect_with_aabb(ray_bundle.origins, ray_bundle.directions, aabb)
        ray_bundle.nears = nears[..., None]
        ray_bundle.fars = fars[..., None]
        ray_bundle.valid_mask = valid_mask[..., None]
        return ray_bundle


class NearFarCollider(SceneBoundsCollider):
    """Sets the nears and fars with fixed values."""

    def __init__(self, near_plane, far_plane, **kwargs) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        super().__init__(**kwargs)

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        ray_bundle.nears = ones * self.near_plane
        ray_bundle.fars = ones * self.far_plane
        ray_bundle.valid_mask = ones.bool()
        return ray_bundle
