"""
Code to handle loading blender datasets.
"""

import os
from typing import Optional

import imageio
import numpy as np
import torch
from torchtyping import TensorType

from mattport.nerf.dataset.structs import DatasetInputs
from mattport.utils.io import load_from_json


def load_blender_data(
    basedir: str, downscale_factor: int = 1, alpha_color: Optional[TensorType[3]] = None, split: str = "train"
) -> DatasetInputs:
    """Processes the a blender dataset directory.
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.

    Args:
        data_directory (str): Location of data
        downscale_factor (int, optional): How much to downscale images. Defaults to 1.0.
        alpha_color (TensorType[3], optional): Sets transparent regions to specified color, otherwise black.
        split (str, optional): Which dataset split to generate.

    Returns:
        DatasetInputs
    """
    # pylint: disable=unused-argument
    # pylint: disable-msg=too-many-locals
    meta = load_from_json(os.path.join(basedir, f"transforms_{split}.json"))
    image_filenames = []
    poses = []
    for frame in meta["frames"]:
        fname = os.path.join(basedir, frame["file_path"].replace("./", "") + ".png")
        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
    poses = np.array(poses).astype(np.float32)

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

    dataset_inputs = DatasetInputs(
        image_filenames=image_filenames,
        downscale_factor=downscale_factor,
        alpha_color=alpha_color,
        intrinsics=intrinsics * 1.0 / downscale_factor,  # downscaling the intrinsics here
        camera_to_world=camera_to_world,
    )

    return dataset_inputs
