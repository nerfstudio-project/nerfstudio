"""
Test to verify disk dataparser
"""

import numpy as np
import torch
from pathlib import Path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from nerfstudio.data.dataparsers.disk_dataparser import (
    DiskDataParser,
    DiskDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox

image_filenames = np.array(["test_image"]).reshape(1, 1)
mask_filenames = np.array(["test_mask"]).reshape(1, 1)
semantics = np.array(["test_semantics"]).reshape(1, 1)
fx = np.array([360]).reshape(1, 1)
fy = np.array([370]).reshape(1, 1)
cx = np.array([500]).reshape(1, 1)
cy = np.array([600]).reshape(1, 1)
distortion_params = np.array([0, 0, 0, 0]).reshape(1, 4)
camera_type = np.array([0]).reshape(1, 1)
camera_to_worlds = np.eye(4).reshape(1, 4, 4)
scene_box = np.array([-1, -1, -1, 1, 1, 1])
height = np.array([1200]).reshape(1, 1)
width = np.array([1000]).reshape(1, 1)

out = {
    "image_filenames": image_filenames,
    "mask_filenames": mask_filenames,
    "semantics": semantics,
    "fx": fx,
    "fy": fy,
    "cx": cx,
    "cy": cy,
    "camera_type": camera_type,
    "camera_to_worlds": camera_to_worlds,
    "scene_box": scene_box,
    "height": height,
    "width": width,
    "distortion_params": distortion_params,
}

np.savez("tests/diskparser/train.npz", **out)

config = DiskDataParserConfig(data=Path("tests/diskparser"))
parser = DiskDataParser(config=config)
dataparser_outputs = parser.get_dataparser_outputs(split="train")
print(dataparser_outputs)