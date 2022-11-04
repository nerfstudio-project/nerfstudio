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

"""Data parser for record3d dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from rich.console import Console
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils import poses as pose_utils
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)


@dataclass
class Record3DDataParserConfig(DataParserConfig):
    """Record3D dataset config"""

    _target: Type = field(default_factory=lambda: Record3D)
    """target class to instantiate"""
    data: Path = Path("data/record3d/bear")
    """Location of data"""
    val_skip: int = 8
    """1/val_skip images to use for validation."""
    aabb_scale: float = 4.0
    """Scene scale."""
    orientation_method: Literal["pca", "up"] = "up"
    """The method to use for orientation"""
    max_dataset_size: int = 300
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""


@dataclass
class Record3D(DataParser):
    """Record3D Dataset"""

    config: Record3DDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:

        CONSOLE.print(
            "[bold red]DEPRECATION WARNING: The Record3D dataparser will be deprecated in future versions. "
            "Use `ns-data-process record3d` to convert the data into the nerfstudio format instead."
        )

        image_dir = self.config.data / "rgb"

        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} doesn't exist")

        image_filenames = []
        for f in image_dir.iterdir():
            if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
                image_filenames.append(f)

        image_filenames = sorted(image_filenames, key=lambda fn: int(fn.stem))
        image_filenames = np.array(image_filenames)
        num_images = len(image_filenames)

        metadata_path = self.config.data / "metadata.json"
        metadata_dict = load_from_json(metadata_path)

        poses_data = np.array(metadata_dict["poses"])
        # (N, 3, 4)
        poses = np.concatenate(
            [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
            axis=-1,
        ).astype(np.float32)

        if self.config.max_dataset_size != -1 and num_images > self.config.max_dataset_size:
            # Evenly select max_dataset_size images from dataset, including first
            # and last indices.
            idx = np.round(np.linspace(0, num_images - 1, self.config.max_dataset_size)).astype(int)
            poses = poses[idx]
            image_filenames = image_filenames[idx]
            num_images = len(image_filenames)

        idx_test = np.arange(num_images)[:: self.config.val_skip]
        idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
        idx = idx_train if split == "train" else idx_test
        if num_images != poses.shape[0]:
            raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

        image_filenames = image_filenames[idx]
        poses = poses[idx]

        # convert to Tensors
        poses = torch.from_numpy(poses[:, :3, :4])

        poses = camera_utils.auto_orient_and_center_poses(
            pose_utils.to4x4(poses), method=self.config.orientation_method
        )[:, :3, :4]

        # Centering poses
        poses[:, :3, 3] = poses[:, :3, 3] - torch.mean(poses[:, :3, 3], dim=0)
        poses = pose_utils.normalize(poses)

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

        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32) * self.config.aabb_scale
        scene_box = SceneBox(aabb=aabb)

        cameras = Cameras(
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_to_worlds=poses,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )

        return dataparser_outputs
