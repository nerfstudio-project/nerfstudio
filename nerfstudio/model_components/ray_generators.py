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
        pose_optimizer: pose optimization module, for optimizing noisy camera intrinsics/extrinsics.
    """

    def __init__(self, cameras: Cameras, pose_optimizer: CameraOptimizer) -> None:
        super().__init__()
        self.cameras = cameras
        self.pose_optimizer = pose_optimizer
        self.register_buffer("image_coords", cameras.get_image_coords(), persistent=False)

    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        camera_opt_to_camera = self.pose_optimizer(c)

        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
            camera_opt_to_camera=camera_opt_to_camera,
        )
        return ray_bundle
