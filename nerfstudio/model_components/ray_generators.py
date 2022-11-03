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
Ray generator.
"""
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
        pose_optimizer: pose optimization module, for optimizing noisy camera intrisics/extrinsics.
    """

    def __init__(self, cameras: Cameras, pose_optimizer: CameraOptimizer, patch_size: int) -> None:
        super().__init__()
        self.cameras = cameras
        self.pose_optimizer = pose_optimizer
        image_coords = cameras.get_image_coords().unsqueeze(0).permute(0, 3, 1, 2)
        self.patch_size=  patch_size
        h, w = image_coords.shape[2:]
        effective_height = h + 1 - patch_size
        effective_width = w + 1 - patch_size
        im2col_values = nn.functional.unfold(image_coords, kernel_size=patch_size, padding=0).permute(0, 2, 1).reshape(effective_height, effective_width, 2, patch_size**2).permute(0, 1, 3, 2)
        self.image_coords = nn.Parameter(im2col_values, requires_grad=False)

    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indicies for target rays.
        """
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x].reshape(-1, 2)
        camera_indices = ray_indices[:, :1].expand(-1, self.patch_size**2).flatten()
        camera_opt_to_camera = self.pose_optimizer(camera_indices)

        ray_bundle = self.cameras.generate_rays(
            camera_indices=camera_indices.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=camera_opt_to_camera,
        )
        return ray_bundle
