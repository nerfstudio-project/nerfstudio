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

import copy
from dataclasses import dataclass
from typing import List, Tuple

import torch
from pyrad.cameras.cameras import Camera, get_camera, get_intrinsics_from_intrinsics_matrix
from pyrad.cameras.utils import get_interpolated_poses_many

import pyrad.cameras.utils as camera_utils


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
    device = camera_a.device
    Ka = camera_a.get_intrinsics_matrix().cpu().numpy()
    pose_a = camera_a.get_camera_to_world_h().cpu().numpy()
    Kb = camera_b.get_intrinsics_matrix().cpu().numpy()
    pose_b = camera_b.get_camera_to_world_h().cpu().numpy()
    poses = [pose_a, pose_b]
    Ks = [Ka, Kb]
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)

    cameras = []
    for pose, K in zip(poses, Ks):
        intrinsics = get_intrinsics_from_intrinsics_matrix(K).to(device).float()
        # TODO: this makes cx and cy an integer, but this code should be fixed
        # it was added to avoid floating point errors when rescaling and the image resolution
        # being different per rendered image in a camera path
        intrinsics[:2] = torch.round(intrinsics[:2])
        camera_to_world = torch.from_numpy(pose[:3]).to(device).float()
        camera = get_camera(intrinsics, camera_to_world, camera_index=camera_a.camera_index)
        cameras.append(camera)
    return CameraPath(cameras=cameras)


def get_spiral_path(
    camera: Camera,
    steps: int = 30,
    radius: float = None,
    radiuses: Tuple[float] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> CameraPath:
    """
    Returns a list of camera in a sprial trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        CameraPath: A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    # TODO: don't hardcode this. pass this in
    up = camera.camera_to_world[:3, 2]  # scene is z up
    # up = camera.camera_to_world[:3, 1] # this will rotate 90 degrees
    focal = min(camera.fx, camera.fy)
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2wh_global = camera.get_camera_to_world_h()

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        ones = torch.tensor([0, 0, 0, 1], device=c2w.device)[None]
        c2wh = torch.cat([c2w, ones], dim=0)
        local_c2whs.append(c2wh)

    cameras = []
    for local_c2wh in local_c2whs:
        cam = copy.deepcopy(camera)
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        cam.camera_to_world = c2wh[:3, :4]
        cameras.append(cam)

    return CameraPath(cameras=cameras)
