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
Code for camera paths.
"""

from typing import List

import torch
from pyrad.cameras.cameras import Camera, get_camera, get_intrinsics_from_intrinsics_matrix
from pyrad.cameras.transformations import get_interpolated_poses, get_interpolated_poses_many
import numpy as np


class CameraPath:
    """
    Base class for camera path.
    """

    def __init__(self) -> None:
        pass


class InterpolatedCameraPath(CameraPath):
    """
    Class to interpolate between two cameras.
    """

    def __init__(self, camera_a: Camera, camera_b: Camera) -> None:
        super().__init__()
        self.camera_a = camera_a
        self.camera_b = camera_b

    def get_path(self, steps: int) -> List[Camera]:
        """
        Returns a list of cameras.
        """
        K_a = self.camera_a.get_intrinsics_matrix()
        pose_a = self.camera_a.get_camera_to_world_h()
        K_b = self.camera_b.get_intrinsics_matrix()
        pose_b = self.camera_b.get_camera_to_world_h()
        poses = [pose_a, pose_b]
        Ks = [K_a, K_b]
        poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)
        # create a list of cameras

        cameras = []
        for pose, K in zip(poses, Ks):
            intrinsics = get_intrinsics_from_intrinsics_matrix(K)
            camera_to_world = torch.from_numpy(pose[:3])
            camera = get_camera(intrinsics, camera_to_world, None)
            cameras.append(camera)
        return cameras
