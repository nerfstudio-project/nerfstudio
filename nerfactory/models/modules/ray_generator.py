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

from nerfactory.cameras.cameras import get_camera_model
from nerfactory.cameras.rays import RayBundle


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class will store the intrinsics and extrinsics parameters of the cameras.

    Args:
        intrinsics: The intrinsics parameters.
        camera_to_world: Camera to world transformation matrix.
        trainable_poses: Whether the c2w matrices can be updated during training.
    """

    def __init__(
        self,
        intrinsics: TensorType["num_cameras", "num_intrinsics_params"],
        camera_to_world: TensorType["num_cameras", 3, 4],
        trainable_poses: bool = False,
    ) -> None:
        super().__init__()
        self.num_cameras, self.num_intrinsics_params = intrinsics.shape
        assert self.num_cameras >= 0
        self.intrinsics = nn.Parameter(intrinsics, requires_grad=False)
        self.camera_to_world = nn.Parameter(camera_to_world, requires_grad=False)
        # TODO(ethan): add learnable parameters that are deltas on the intrinsics and camera_to_world parameters
        self.trainable_poses = trainable_poses
        self.camera_to_world_delta = nn.Parameter(torch.zeros_like(camera_to_world), requires_grad=self.trainable_poses)

        # NOTE(ethan): we currently assume all images have the same height and width
        camera_index = 0
        self.camera_class = get_camera_model(self.num_intrinsics_params)
        camera = self.camera_class(*self.intrinsics[camera_index].tolist())
        self.image_coords = nn.Parameter(camera.get_image_coords(), requires_grad=False)
        self.register_full_backward_hook(self.backward_hook)

    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indicies for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        intrinsics = self.intrinsics[c]
        camera_to_world = self.camera_to_world[c]
        camera_to_world += self.camera_to_world_delta[c] # will be a no-op if no trainable poses.
        coords = self.image_coords[y, x]

        ray_bundle = self.camera_class.generate_rays(
            intrinsics=intrinsics, camera_to_world=camera_to_world, coords=coords
        )
        ray_bundle.camera_indices = c[..., None]  # ["num_rays",1]
        return ray_bundle

    def backward_hook(self, module, grad_input, grad_output):
        print("ALEX:", grad_input, grad_output)
        # raise AssertionError("IT WORKS!")
