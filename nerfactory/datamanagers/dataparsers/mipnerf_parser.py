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

"""Data parser for mipnerf dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfactory.cameras.cameras import Cameras, CameraType
from nerfactory.datamanagers.dataparsers.base import DataParser, DataParserConfig
from nerfactory.datamanagers.structs import DatasetInputs, SceneBounds
from nerfactory.utils.io import get_absolute_path


@dataclass
class MipNerf360DataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset parser config"""

    _target: Type = field(default_factory=lambda: Mipnerf360)
    """target class to instantiate"""
    data_directory: Path = Path("data/mipnerf_360/garden")
    """directory specifying location of data"""
    downscale_factor: int = 1
    """How much to downscale images."""
    val_skip: int = 8
    """1/val_skip images to use for validation."""
    auto_scale: bool = True
    """Scale based on pose bounds."""
    aabb_scale: float = 4
    """Scene scale."""


@dataclass
class Mipnerf360(DataParser):
    """MipNeRF 360 Dataset"""

    config: MipNerf360DataParserConfig

    @classmethod
    def normalize_orientation(cls, poses: np.ndarray):
        """Set the _up_ direction to be in the positive Y direction.

        Args:
            poses: Numpy array of poses.
        """
        poses_orig = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        center = poses[:, :3, 3].mean(0)
        vec2 = poses[:, :3, 2].sum(0) / np.linalg.norm(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        vec0 = np.cross(up, vec2) / np.linalg.norm(np.cross(up, vec2))
        vec1 = np.cross(vec2, vec0) / np.linalg.norm(np.cross(vec2, vec0))
        c2w = np.stack([vec0, vec1, vec2, center], -1)  # [3, 4]
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # [4, 4]
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # [BS, 1, 4]
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # [BS, 4, 4]
        poses = np.linalg.inv(c2w) @ poses
        poses_orig[:, :3, :4] = poses[:, :3, :4]
        return poses_orig

    def _generate_dataset_inputs(self, split="train"):
        abs_dir = get_absolute_path(self.config.data_directory)
        image_dir = "images"
        if self.config.downscale_factor > 1:
            image_dir += f"_{self.config.downscale_factor}"
        image_dir = abs_dir / image_dir
        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} doesn't exist")

        valid_formats = [".jpg", ".png"]
        image_filenames = []
        for f in image_dir.iterdir():
            ext = f.suffix
            if ext.lower() not in valid_formats:
                continue
            image_filenames.append(image_dir / f)
        image_filenames = sorted(image_filenames)
        num_images = len(image_filenames)

        poses_data = np.load(abs_dir / "poses_bounds.npy")
        poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
        bounds = poses_data[:, -2:].transpose([1, 0])

        if num_images != poses.shape[0]:
            raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

        idx_test = np.arange(num_images)[:: self.config.val_skip]
        idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
        idx = idx_train if split == "train" else idx_test

        image_filenames = np.array(image_filenames)[idx]
        poses = poses[idx]

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]

        poses[:, :2, 4] = np.array([image_height, image_width])
        poses[:, 2, 4] = poses[:, 2, 4]

        # Reorder pose to match our convention
        poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

        # Center poses and rotate. (Compute up from average of all poses)
        poses = self.normalize_orientation(poses)

        # Scale factor used in mipnerf
        if self.config.auto_scale:
            scale_factor = 1 / (np.min(bounds) * 0.75)
            poses[:, :3, 3] *= scale_factor
            bounds *= scale_factor

        # Center poses
        poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, :], axis=0)[:, 3]

        focal_length = poses[0, -1, -1]

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3, :4])  # camera to world transform
        num_cameras = len(image_filenames)
        num_intrinsics_params = 3
        intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
        intrinsics *= torch.tensor([cx, cy, focal_length])

        aabb = torch.tensor([[-4, -4, -4], [4, 4, 4]], dtype=torch.float32) * self.config.aabb_scale
        scene_bounds = SceneBounds(aabb=aabb)

        cameras = Cameras(
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_bounds=scene_bounds,
        )

        return dataset_inputs
