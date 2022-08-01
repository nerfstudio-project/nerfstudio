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
Code for loading Record3D formats from iPhone app.

credit: Hang Gao (hangg@berkeley.edu) for the pose extraction code.
"""

import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from nerfactory.data.structs import DatasetInputs


def load_record_3d_data(
    basedir: str,
    downscale_factor: int = 1,
    split: str = "train",
    val_skip: int = 8,
    auto_scale: bool = True,
) -> DatasetInputs:
    """Processes Record3D data.

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

    image_dir = os.path.join(basedir, "rgb")

    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory {image_dir} doesn't exist")

    ext = ".jpg"
    image_filenames = []
    for f in os.listdir(image_dir):
        ext = os.path.splitext(f)[1]
        image_filenames.append(os.path.join(image_dir, f))
    image_filenames = sorted(image_filenames, key=lambda fn: int(fn[: len(ext)]))
    num_images = len(image_filenames)

    metadata_path = os.path.join(basedir, "metadata.json")

    with open(str(metadata_path)) as f:
        metadata_dict = json.load(f)

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    focal_length = K[0, 0]
    cx, cy = K[:2, -1]

    H = metadata_dict["h"]
    W = metadata_dict["w"]

    num_cameras = 1
    num_intrinsics_params = 3
    intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
    intrinsics *= torch.tensor([cx, cy, focal_length])

    poses_data = metadata_dict["poses"]
    # (N, 3, 4)
    poses = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)

    idx_test = np.arange(num_images)[::val_skip]
    idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
    idx = idx_train if split == "train" else idx_test

    poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
    bounds = poses_data[:, -2:].transpose([1, 0])

    if num_images != poses.shape[0]:
        raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

    idx_test = np.arange(num_images)[::val_skip]
    idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
    idx = idx_train if split == "train" else idx_test

    image_filenames = np.array(image_filenames)[idx]
    poses = poses[idx]

    poses[:, :2, 4] = np.array([H, W])
    poses[:, 2, 4] = poses[:, 2, 4] * 1.0 / downscale_factor

    # Reorder pose to match our convention
    # poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

    camera_to_world = torch.from_numpy(poses[:, :3, :4])  # camera to world transform
    # Scale factor used in mipnerf
    if auto_scale:
        scale_factor = 1 / (np.min(bounds) * 0.75)
        poses[:, :3, 3] *= scale_factor
        bounds *= scale_factor

    dataset_inputs = DatasetInputs(
        image_filenames=image_filenames,
        downscale_factor=1,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )

    return dataset_inputs
