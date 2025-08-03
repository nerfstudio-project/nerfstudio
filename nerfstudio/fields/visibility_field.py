# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Visibility Field"""

import torch
from torch import nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples


class VisibilityField(nn.Module):
    """Visibility Field"""

    def __init__(self, cameras: Cameras) -> None:
        super().__init__()
        # training camera tranforms
        # TODO: use optimized cameras
        self.c2ws = cameras.camera_to_worlds
        self.c2whs = torch.cat([self.c2ws, torch.zeros_like(self.c2ws[:, :1, :])], dim=1)
        self.c2whs[:, 3, 3] = 1.0
        self.w2chs = torch.inverse(self.c2whs)
        self.K = cameras.get_intrinsics_matrices()
        self.image_height = cameras.height
        self.image_width = cameras.width

    @torch.no_grad()
    def forward(self, ray_samples: RaySamples, camera_chunk_size=50, ray_chunk_size=4096) -> torch.Tensor:
        """
        Args:
            ray_samples: Ray samples.
            camera_chunk_size: Number of cameras to process at once to avoid memory issues.
            ray_chunk_size: Number of rays to process at once to avoid memory issues.
        Returns:
        """
        # get positions
        positions = ray_samples.frustums.get_positions()  # [N, S, 3]
        # project positions into each camera
        # move to homogeneous coordinates
        positions = torch.cat([positions, torch.ones_like(positions[..., :1])], dim=-1)
        N, S, _ = positions.shape
        B = self.w2chs.shape[0]  # num cameras
        p = positions.view(N * S, 4).transpose(0, 1).unsqueeze(0)  # [1, 4, N*S]
        p = p.expand(B, *p.shape[1:])  # [B, 4, N*S]

        num_views = torch.zeros([N, S, 1], device=positions.device)
        for i in range(0, B, camera_chunk_size):
            ccs = min(camera_chunk_size, B - i)
            for j in range(0, N, ray_chunk_size):
                rcs = min(ray_chunk_size, N - j)

                ptemp = p.reshape(B, 4, N, S)[i : i + ccs, :, j : j + rcs, :].reshape(ccs, 4, rcs * S)
                cam_coords = torch.bmm(self.w2chs[i : i + ccs, :], ptemp)

                # flip y and z axes
                cam_coords[:, 1, :] *= -1
                cam_coords[:, 2, :] *= -1

                z = cam_coords[:, 2:3, :].transpose(1, 2).view(ccs, rcs, S, 1)  # [CS, RCS, S, 1]
                mask_z = z[..., 0] > 0

                # divide by z
                cam_coords = cam_coords[:, :3, :] / cam_coords[:, 2:3, :]

                cam_points = torch.bmm(self.K[i : i + ccs], cam_coords)

                pixel_coords = cam_points[:, :2, :].transpose(1, 2).view(ccs, rcs, S, 2)  # [CS, RCS, S, 2]
                x = pixel_coords[..., 0]
                y = pixel_coords[..., 1]
                mask_x = (x >= 0) & (x < self.image_width.view(B, 1, 1)[i : i + ccs])
                mask_y = (y >= 0) & (y < self.image_height.view(B, 1, 1)[i : i + ccs])
                mask = mask_x & mask_y & mask_z
                # sum over the batch dimension
                nv = mask.sum(dim=0).unsqueeze(-1)
                # nv is [N, S, 1] # this is the number of camera frustums that the point belongs to
                # for this particular chunk of cameras
                num_views[j : j + rcs] += nv
        return num_views
