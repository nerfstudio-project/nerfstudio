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
Code for loading instant-ngp dataset formats.
# TODO
"""

import logging
import os

import imageio
import numpy as np
import torch

from pyrad.data.structs import DatasetInputs, SceneBounds
from pyrad.utils.io import load_from_json


def load_instant_ngp_data(
    basedir: str, downscale_factor: int = 1, split: str = "train", camera_translation_scalar=0.33
) -> DatasetInputs:
    """Processes the a blender dataset directory.
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.

    Args:
        data_directory (str): Location of data
        downscale_factor (int, optional): How much to downscale images. Defaults to 1.0.
        split (str, optional): Which dataset split to generate.

    Returns:
        DatasetInputs
    """
    # pylint: disable=unused-argument
    # pylint: disable-msg=too-many-locals
    meta = load_from_json(os.path.join(basedir, "transforms.json"))
    image_filenames = []
    poses = []
    num_skipped_image_filenames = 0
    for frame in meta["frames"]:
        fname = os.path.join(basedir, frame["file_path"])
        if not os.path.exists(fname):
            num_skipped_image_filenames += 1
        else:
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
    if num_skipped_image_filenames >= 0:
        logging.info("Skipping %s files in dataset split %s.", num_skipped_image_filenames, split)
    assert (
        len(image_filenames) != 0
    ), """
    No image files found. 
    You should check the file_paths in the transforms.json file to make sure they are correct.
    """
    poses = np.array(poses).astype(np.float32)
    poses[:3, 3] *= camera_translation_scalar

    img_0 = imageio.imread(image_filenames[0])
    image_height, image_width = img_0.shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    cx = image_width / 2.0
    cy = image_height / 2.0
    camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
    num_cameras = len(image_filenames)
    num_intrinsics_params = 3
    intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
    intrinsics *= torch.tensor([cx, cy, focal_length])

    # in x,y,z order
    # assumes that the scene is centered at the origin
    aabb_scale = meta["aabb_scale"]
    scene_bounds = SceneBounds(
        aabb=torch.tensor(
            [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
        )
    )

    # TODO(ethan): add alpha background color
    dataset_inputs = DatasetInputs(
        image_filenames=image_filenames,
        downscale_factor=downscale_factor,
        intrinsics=intrinsics * 1.0 / downscale_factor,  # downscaling the intrinsics here
        camera_to_world=camera_to_world,
        scene_bounds=scene_bounds,
    )

    return dataset_inputs
