"""
For loading the blender dataset format.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import imageio
import numpy as np
import torch

from mattport.utils.io import get_absolute_path, load_from_json


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
        intrinsics=intrinsics * 1.0 / downscale_factor,  # downscaling the intrinsics here
        camera_to_world=camera_to_world,
    )

    return dataset_inputs


def get_dataset_inputs_dict(
    data_directory: str, dataset_type: str, downscale_factor: float = 1.0, splits: Tuple[str] = ("train", "val")
) -> Dict[str, DatasetInputs]:
    """Returns the dataset inputs, which will be used with an ImageDataset and RayGenerator.
    # TODO: implement the `test` split, which will have depths and normals, etc.

    Args:
        data_directory (str): _description_
        dataset_type (str): _description_
        downscale_factor (float, optional): _description_. Defaults to 1.0.

    Returns:
        Dict[str, DatasetInputs]: The inputs needed for generating rays.
    """
    dataset_inputs_dict = {}
    if dataset_type == "blender":
        for split in splits:
            dataset_inputs = load_blender_data(get_absolute_path(data_directory), downscale_factor=downscale_factor)
            dataset_inputs_dict[split] = dataset_inputs
    else:
        raise NotImplementedError()

    return dataset_inputs_dict
