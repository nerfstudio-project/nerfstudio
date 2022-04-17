"""
For loading the blender dataset format.
"""

import os
from dataclasses import dataclass
from typing import List

import imageio
import numpy as np
import torch

from mattport.utils.io import load_from_json


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        return self[attr]

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@dataclass
class DatasetInputs:
    """Dataset inputs for the image dataset and the ray generator."""

    image_filenames: List[str]
    downscale_factor: float = 1.0
    intrinsics: torch.tensor = None
    camera_to_world: torch.tensor = None


def load_blender_data(basedir, downscale_factor=1.0, split="train"):
    """Processes the a blender dataset directory.
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.

    Args:
        basedir (_type_): _description_
        downscale_factor (bool, optional): _description_. Defaults to False.
        testskip (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
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
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )

    return dataset_inputs


def get_dataset_inputs(data_directory: str, dataset_type: str, downscale_factor: float = 1.0):
    """Returns the dataset inputs, which will be used with an ImageDataset and RayGenerator.

    Args:
        data_directory (str): _description_
        dataset_type (str): _description_
        downscale_factor (float, optional): _description_. Defaults to 1.0.

    Returns:
        DatasetInputs: The inputs needed for generating rays.
    """
    if dataset_type == "blender":
        dataset_inputs = load_blender_data(data_directory, downscale_factor=downscale_factor)
    else:
        raise NotImplementedError()

    return dataset_inputs
