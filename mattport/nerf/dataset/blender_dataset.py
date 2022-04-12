"""
For loading the blender dataset format.
"""

import os

import imageio
import numpy as np
from mattport.utils.io import load_from_json


def load_blender_data(basedir, half_res=False, testskip=1):
    """Processes the a blender dataset directory.
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.

    Args:
        basedir (_type_): _description_
        half_res (bool, optional): _description_. Defaults to False.
        testskip (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # pylint: disable=unused-argument
    # pylint: disable-msg=too-many-locals
    splits = ["train", "val", "test"]
    metas = {}
    for split in splits:
        metas[split] = load_from_json(os.path.join(basedir, f"transforms_{split}.json"))

    image_filenames = []
    all_poses = []
    for split in splits:
        meta = metas[split]
        poses = []
        if split == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        all_poses.append(poses)

    poses = np.concatenate(all_poses, 0)

    img_0 = imageio.imread(image_filenames[0])
    H, W = img_0.shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    return image_filenames, poses, focal, H, W
