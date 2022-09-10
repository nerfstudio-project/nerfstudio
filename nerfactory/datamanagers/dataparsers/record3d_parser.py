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

"""Data parser for record3d dataset"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from nerfactory.cameras.cameras import Cameras, CameraType
from nerfactory.configs import base as cfg
from nerfactory.datamanagers.dataparsers.base import DataParser
from nerfactory.datamanagers.dataparsers.mipnerf_parser import Mipnerf360
from nerfactory.datamanagers.structs import DatasetInputs, SceneBounds
from nerfactory.utils.io import get_absolute_path, load_from_json


@dataclass
class Record3D(DataParser):
    """Record3D Dataset

    Args:
        data_directory: Location of data
        val_skip: 1/val_skip images to use for validation. Defaults to 8.
        aabb_scale: Scene scale, Defaults to 4.0.
        max_dataset_size: Max number of images to train on. If the dataset has
            more, images will be sampled approximately evenly. Defaults to 150.
    """

    config: cfg.Record3DDataParserConfig

    def _generate_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        abs_dir = get_absolute_path(self.config.data_directory)

        image_dir = abs_dir / "rgb"

        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} doesn't exist")

        image_filenames = []
        for f in image_dir.iterdir():
            if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
                image_filenames.append(image_dir / f)

        image_filenames = sorted(image_filenames, key=lambda fn: int(fn.stem))
        image_filenames = np.array(image_filenames)
        num_images = len(image_filenames)

        metadata_path = abs_dir / "metadata.json"
        metadata_dict = load_from_json(metadata_path)

        poses_data = np.array(metadata_dict["poses"])
        # (N, 3, 4)
        poses = np.concatenate(
            [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
            axis=-1,
        ).astype(np.float32)

        if num_images > self.config.max_dataset_size:
            # Evenly select max_dataset_size images from dataset, including first
            # and last indices.
            idx = np.round(np.linspace(0, num_images - 1, self.config.max_dataset_size)).astype(int)
            poses = poses[idx]
            image_filenames = image_filenames[idx]
            num_images = len(image_filenames)

        # Normalization similar to Mipnerf360
        poses = Mipnerf360.normalize_orientation(poses)

        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2).astype(np.float32)

        rotation_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        poses = rotation_matrix @ poses

        idx_test = np.arange(num_images)[:: self.config.val_skip]
        idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
        idx = idx_train if split == "train" else idx_test
        if num_images != poses.shape[0]:
            raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

        image_filenames = image_filenames[idx]
        poses = poses[idx]

        # Centering poses
        poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, 3], axis=0)

        camera_to_world = torch.from_numpy(poses[:, :3, :4])  # camera to world transform

        # Camera intrinsics
        K = np.array(metadata_dict["K"]).reshape((3, 3)).T
        focal_length = K[0, 0]

        H = metadata_dict["h"]
        W = metadata_dict["w"]

        # TODO(akristoffersen): The metadata dict comes with principle points,
        # but caused errors in image coord indexing. Should update once that is fixed.
        cx, cy = W / 2, H / 2

        num_cameras = len(image_filenames)
        num_intrinsics_params = 3
        intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
        intrinsics *= torch.tensor([cx, cy, focal_length])

        # scene_bounds = SceneBounds.from_camera_poses(camera_to_world, self.config.aabb_scale)
        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32) * self.config.aabb_scale
        scene_bounds = SceneBounds(aabb=aabb)

        cameras = Cameras(
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_bounds=scene_bounds,
        )

        return dataset_inputs
