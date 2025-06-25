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

"""Data parser for ARKitScenes dataset"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


# Taken from https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
def traj_string_to_matrix(traj_string: str):
    """convert traj_string into translation and rotation matrices
    Args:
        traj_string: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)
    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    tokens = traj_string.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))  # type: ignore
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


@dataclass
class ARKitScenesDataParserConfig(DataParserConfig):
    """ARKitScenes dataset config.
    ARKitScenes dataset (http://github.com/apple/ARKitScenes) is a large-scale 3D dataset of indoor scenes.
    This dataparser uses 3D detection subset of the ARKitScenes dataset.
    """

    _target: Type = field(default_factory=lambda: ARKitScenes)
    """target class to instantiate"""
    data: Path = Path("data/ARKitScenes/3dod/Validation/41069021")
    """Path to ARKitScenes folder with densely extracted scenes."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""


@dataclass
class ARKitScenes(DataParser):
    """ARKitScenes DatasetParser"""

    config: ARKitScenesDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        video_id = self.config.data.name

        image_dir = self.config.data / f"{video_id}_frames" / "lowres_wide"
        depth_dir = self.config.data / f"{video_id}_frames" / "lowres_depth"
        intrinsics_dir = self.config.data / f"{video_id}_frames" / "lowres_wide_intrinsics"
        pose_file = self.config.data / f"{video_id}_frames" / "lowres_wide.traj"

        frame_ids = [x.name for x in sorted(depth_dir.iterdir())]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        frame_ids.sort()

        poses_from_traj = {}
        with open(pose_file, "r", encoding="utf-8") as f:
            traj = f.readlines()

        for line in traj:
            poses_from_traj[f"{round(float(line.split(' ')[0]), 3):.3f}"] = np.array(
                traj_string_to_matrix(line)[1].tolist()
            )

        image_filenames, depth_filenames, intrinsics, poses = [], [], [], []
        w, h, _, _, _, _ = np.loadtxt(list(sorted(intrinsics_dir.iterdir()))[0])  # Get image size from first intrinsic

        for frame_id in frame_ids:
            intrinsic = self._get_intrinsic(intrinsics_dir, frame_id, video_id)
            frame_pose = self._get_pose(frame_id, poses_from_traj)

            intrinsics.append(intrinsic)
            image_filenames.append(image_dir / f"{video_id}_{frame_id}.png")
            depth_filenames.append(depth_dir / f"{video_id}_{frame_id}.png")
            poses.append(frame_pose)

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.config.train_split_fraction)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))

        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method="none",
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        intrinsics = intrinsics[indices.tolist()]
        poses = poses[indices.tolist()]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=int(h),
            width=int(w),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs

    @staticmethod
    def _get_intrinsic(intrinsics_dir: Path, frame_id: str, video_id: str):
        intrinsic_fn = intrinsics_dir / f"{video_id}_{frame_id}.pincam"

        if not intrinsic_fn.exists():
            intrinsic_fn = intrinsics_dir / f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam"

        if not intrinsic_fn.exists():
            intrinsic_fn = intrinsics_dir / f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam"

        _, _, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)
        intrinsic = np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
        return intrinsic

    @staticmethod
    def _get_pose(frame_id: str, poses_from_traj: dict):
        frame_pose = None
        if str(frame_id) in poses_from_traj:
            frame_pose = np.array(poses_from_traj[str(frame_id)])
        else:
            for my_key in poses_from_traj:
                if abs(float(frame_id) - float(my_key)) < 0.005:
                    frame_pose = np.array(poses_from_traj[str(my_key)])

        assert frame_pose is not None
        frame_pose[0:3, 1:3] *= -1
        frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
        frame_pose[2, :] *= -1
        return frame_pose
