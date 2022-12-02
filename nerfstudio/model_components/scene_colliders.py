# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Scene Colliders
"""

from __future__ import annotations

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox


class SceneCollider(nn.Module):
    """Module for setting near and far values for rays."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        super().__init__()

    def set_nears_and_fars(self, ray_bundle) -> RayBundle:
        """To be implemented."""
        raise NotImplementedError

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        """Sets the nears and fars if they are not set already."""
        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            return ray_bundle
        return self.set_nears_and_fars(ray_bundle)


class AABBBoxCollider(SceneCollider):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, scene_box: SceneBox, near_plane: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scene_box = scene_box
        self.near_plane = near_plane

    def _intersect_with_aabb(
        self, rays_o: TensorType["num_rays", 3], rays_d: TensorType["num_rays", 3], aabb: TensorType[2, 3]
    ):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins
            rays_d: (num_rays, 3) ray directions
            aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
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

        # clamp to near plane
        near_plane = self.near_plane if self.training else 0
        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)

        return nears, fars

    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.

        Args:
            ray_bundle: specified ray bundle to operate on
        """
        aabb = self.scene_box.aabb
        nears, fars = self._intersect_with_aabb(ray_bundle.origins, ray_bundle.directions, aabb)
        ray_bundle.nears = nears[..., None]
        ray_bundle.fars = fars[..., None]
        return ray_bundle


class NearFarCollider(SceneCollider):
    """Sets the nears and fars with fixed values.

    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
    """

    def __init__(self, near_plane: float, far_plane: float, **kwargs) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        super().__init__(**kwargs)

    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        near_plane = self.near_plane if self.training else 0
        ray_bundle.nears = ones * near_plane
        ray_bundle.fars = ones * self.far_plane
        return ray_bundle
