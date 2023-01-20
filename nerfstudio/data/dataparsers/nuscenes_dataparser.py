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

"""Data parser for NuScenes dataset"""
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.process_data.colmap_utils import qvec2rotmat


def rotation_translation_to_pose(r_vec, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)
    pose[:3, :3] = qvec2rotmat(r_vec)
    pose[:3, 3] = t_vec
    return pose


@dataclass
class NuScenesDataParserConfig(DataParserConfig):
    """NuScenes dataset config.
    NuScenes (https://www.nuscenes.org/nuscenes) is an autonomous driving dataset containing 1000 20s clips.
    Each clip was recorded with a suite of sensors including 6 surround cameras.
    It also includes 3D cuboid annotations around objects.
    We optionally use these cuboids to mask dynamic objects by specifying the mask_dir flag.
    To create these masks use scripts/datasets/process_nuscenes_masks.py.
    """

    _target: Type = field(default_factory=lambda: NuScenes)
    """target class to instantiate"""
    data: Path = Path("scene-0103")  # TODO: rename to scene but keep checkpoint saving name?
    """Name of the scene."""
    data_dir: Path = Path("/mnt/local/NuScenes")
    """Path to NuScenes dataset."""
    version: Literal["v1.0-mini", "v1.0-trainval"] = "v1.0-mini"
    """Dataset version."""
    cameras: Tuple[Literal["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"], ...] = ("FRONT",)
    """Which cameras to use."""
    mask_dir: Optional[Path] = None
    """Path to masks of dynamic objects."""

    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""

    verbose: bool = False
    """Load dataset with verbose messaging"""


@dataclass
class NuScenes(DataParser):
    """NuScenes DatasetParser"""

    config: NuScenesDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        nusc = NuScenesDatabase(version=self.config.version, dataroot=self.config.data_dir, verbose=self.config.verbose)
        cameras = ["CAM_" + camera for camera in self.config.cameras]

        assert (
            len(cameras) == 1
        ), "waiting on multiple camera support"  # TODO: remove once multiple cameras are supported

        # get samples for scene
        samples = [
            samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] == str(self.config.data)
        ]

        # sort by timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        transform1 = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        transform2 = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )

        # get image filenames and camera data
        image_filenames = []
        mask_filenames = []
        mask_dir = self.config.mask_dir if self.config.mask_dir is not None else Path("")
        intrinsics = []
        poses = []
        for sample in samples:
            for camera in cameras:
                camera_data = nusc.get("sample_data", sample["data"][camera])
                calibrated_sensor_data = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
                ego_pose_data = nusc.get("ego_pose", camera_data["ego_pose_token"])

                ego_pose = rotation_translation_to_pose(ego_pose_data["rotation"], ego_pose_data["translation"])
                cam_pose = rotation_translation_to_pose(
                    calibrated_sensor_data["rotation"], calibrated_sensor_data["translation"]
                )
                pose = ego_pose @ cam_pose

                # rotate to opencv frame
                pose = transform1 @ pose

                # convert from opencv camera to nerfstudio camera
                pose[0:3, 1:3] *= -1
                pose = pose[np.array([1, 0, 2, 3]), :]
                pose[2, :] *= -1

                # rotate to z-up in viewer
                pose = transform2 @ pose

                image_filenames.append(self.config.data_dir / camera_data["filename"])
                mask_filenames.append(
                    mask_dir / "masks" / camera / os.path.split(camera_data["filename"])[1].replace("jpg", "png")
                )
                intrinsics.append(calibrated_sensor_data["camera_intrinsic"])
                poses.append(pose)
        poses = torch.from_numpy(np.stack(poses).astype(np.float32))
        intrinsics = torch.from_numpy(np.array(intrinsics).astype(np.float32))

        # center poses
        poses[:, :3, 3] -= poses[:, :3, 3].mean(dim=0)

        # scale poses
        poses[:, :3, 3] /= poses[:, :3, 3].abs().max()

        # filter image_filenames and poses based on train/eval split percentage
        num_snapshots = len(samples)
        num_train_snapshots = math.ceil(num_snapshots * self.config.train_split_percentage)
        num_eval_snapshots = num_snapshots - num_train_snapshots
        i_all = np.arange(num_snapshots)
        i_train = np.linspace(
            0, num_snapshots - 1, num_train_snapshots, dtype=int
        )  # equally spaced training snapshots starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_snapshots
        i_train = (i_train[None, :] * len(cameras) + np.arange(len(cameras))[:, None]).ravel()
        i_eval = (i_eval[None, :] * len(cameras) + np.arange(len(cameras))[:, None]).ravel()
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices]
        intrinsics = intrinsics[indices]
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 1.0
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
            height=900,
            width=1600,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if self.config.mask_dir is not None else None,
        )
        return dataparser_outputs
