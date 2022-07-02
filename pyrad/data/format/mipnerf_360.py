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
Code for loading mipnerf 360 dataset formats.
"""

import os

import imageio
import numpy as np
import torch

from pyrad.data.structs import DatasetInputs


def load_mipnerf_360_data(
    basedir: str,
    downscale_factor: int = 1,
    split: str = "train",
    val_skip: int = 8,
    auto_scale: bool = True,
) -> DatasetInputs:
    """Processes mipnerf 360 data.
    Example data can be downloaded from https://jonbarron.info/mipnerf360/.

    TODO: code currently assumes images were previously downscaled.

    Args:
        data_directory (str): Location of data
        downscale_factor (int, optional): How much to downscale images. Must be a factor of 2. Defaults to 1.
        split (str, optional): Which dataset split to generate.
        val_skip (int, optional): 1/val_skip images to use for validation. Defaults to 8.
        auto_scale (boolean, optional): Scale based on pose bounds. Defaults to True.
        aabb_scale (float, optional): Scene scale, Defaults to 1.0.

    Returns:
        DatasetInputs
    """

    image_dir = os.path.join(basedir, "images")
    if downscale_factor > 1:
        image_dir += f"_{downscale_factor}"

    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory {image_dir} doesn't exist")

    valid_formats = [".jpg", ".png"]
    image_filenames = []
    for f in os.listdir(image_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_formats:
            continue
        image_filenames.append(os.path.join(image_dir, f))
    image_filenames = sorted(image_filenames)
    num_images = len(image_filenames)

    poses_data = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
    bounds = poses_data[:, -2:].transpose([1, 0])

    if num_images != poses.shape[0]:
        raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

    idx_test = np.arange(num_images)[::val_skip]
    idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
    idx = idx_train if split == "train" else idx_test

    image_filenames = np.array(image_filenames)[idx]
    poses = poses[idx]

    img_0 = imageio.imread(image_filenames[0])
    image_height, image_width = img_0.shape[:2]

    poses[:, :2, 4] = np.array([image_height, image_width])
    poses[:, 2, 4] = poses[:, 2, 4] * 1.0 / downscale_factor

    # Reorder pose to match our convention
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

    # Scale factor used in mipnerf
    if auto_scale:
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

    dataset_inputs = DatasetInputs(
        image_filenames=image_filenames,
        downscale_factor=1,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )

    return dataset_inputs
