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

from dataclasses import dataclass
from typing import List

import torch
from pyrad.cameras.cameras import Camera, get_camera, get_intrinsics_from_intrinsics_matrix
from pyrad.cameras.transformations import get_interpolated_poses_many


@dataclass
class CameraPath:
    """A camera path."""

    cameras: List[Camera]


def get_interpolated_camera_path(camera_a: Camera, camera_b: Camera, steps: int) -> CameraPath:
    """Generate a camera path between two cameras.

    Args:
        camera_a: The first camera.
        camera_b: The second camera.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        CameraPath: A camera path.
    """
    Ka = camera_a.get_intrinsics_matrix()
    pose_a = camera_a.get_camera_to_world_h()
    Kb = camera_b.get_intrinsics_matrix()
    pose_b = camera_b.get_camera_to_world_h()
    poses = [pose_a, pose_b]
    Ks = [Ka, Kb]
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)
    # create a list of cameras

    cameras = []
    for pose, K in zip(poses, Ks):
        intrinsics = get_intrinsics_from_intrinsics_matrix(K)
        camera_to_world = torch.from_numpy(pose[:3])
        camera = get_camera(intrinsics, camera_to_world, None)
        cameras.append(camera)
    return CameraPath(cameras=cameras)


def get_spiral_path(camera_a: Camera, camera_b: Camera, steps: int) -> CameraPath:
    """
    Returns a list of camera in a sprial.
    """
    raise NotImplementedError
