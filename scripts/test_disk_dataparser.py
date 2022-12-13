"""
Test to verify disk dataparser
"""

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

# from nerfstudio.data.dataparsers.disk_dataparser import (
#     DiskDataParser,
#     DiskDataParserConfig,
# )
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

np.savez("out.npz", **out)
data = np.load("out.npz")
print(data)
print(type(data))
print(data.keys())
# with open(self.config.data / split, "r") as f:
#     data = np.load(f)

image_filenames = data["image_filenames"].tolist()
mask_filenames = data["mask_filenames"].tolist() if "mask_filenames" in data.keys() else None
metadata = {"semantics": data["semantic_filenames"].tolist()} if "semantic_filenames" in data.keys() else None
times = data["times"].tolist() if "times" in data.keys() else None

fx = torch.from_numpy(data["fx"])
fy = torch.from_numpy(data["fy"])
cx = torch.from_numpy(data["cx"])
cy = torch.from_numpy(data["cy"])
distortion_params = torch.from_numpy(data["distortion_params"])
camera_to_worlds = torch.from_numpy(data["camera_to_worlds"])
camera_type = torch.from_numpy(data["camera_type"])
height = torch.from_numpy(data["height"])
width = torch.from_numpy(data["width"])
scene_box_aabb = torch.from_numpy(data["scene_box"])

scene_box = SceneBox(aabb=scene_box_aabb)

cameras = Cameras(
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    distortion_params=distortion_params,
    height=height,
    width=width,
    camera_to_worlds=camera_to_worlds[:, :3, :4],
    camera_type=camera_type,
    times=times,
)

dataparser_outputs = DataparserOutputs(
    image_filenames=image_filenames,
    cameras=cameras,
    scene_box=scene_box,
    mask_filenames=mask_filenames,
    metadata=metadata,
)
# return dataparser_outputs
