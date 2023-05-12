# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Data parser for DyCheck (https://arxiv.org/abs/2210.13445) dataset of `iphone` subset"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type

import cv2
import numpy as np
import torch
from rich.progress import Console

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)


def downscale(img, scale: int) -> np.ndarray:
    """Function from DyCheck's repo. Downscale an image.

    Args:
        img: Input image
        scale: Factor of the scale

    Returns:
        New image
    """
    if scale == 1:
        return img
    height, width = img.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f"Image shape ({height},{width}) must be divisible by the" f" scale ({scale}).")
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)
    return resized


def upscale(img, scale: int) -> np.ndarray:
    """Function from DyCheck's repo. Upscale an image.

    Args:
        img: Input image
        scale: Factor of the scale

    Returns:
        New image
    """
    if scale == 1:
        return img
    height, width = img.shape[:2]
    out_height, out_width = height * scale, width * scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)
    return resized


def rescale(img, scale_factor: float, interpolation=cv2.INTER_AREA) -> np.ndarray:
    """Function from DyCheck's repo. Rescale an image.

    Args:
        img: Input image
        scale: Factor of the scale
        interpolation: Interpolation method in opencv

    Returns:
        New image
    """
    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be a non-negative number.")
    if scale_factor == 1.0:
        return img

    height, width = img.shape[:2]
    if scale_factor.is_integer():
        return upscale(img, int(scale_factor))

    inv_scale = 1.0 / scale_factor
    if inv_scale.is_integer() and (scale_factor * height).is_integer() and (scale_factor * width).is_integer():
        return downscale(img, int(inv_scale))

    print(f"Resizing image by non-integer factor {scale_factor}, this may lead to artifacts.")
    height, width = img.shape[:2]
    out_height = math.ceil(height * scale_factor)
    out_height -= out_height % 2
    out_width = math.ceil(width * scale_factor)
    out_width -= out_width % 2

    return cv2.resize(img, (out_width, out_height), interpolation)


def _load_scene_info(data_dir: Path) -> Tuple[np.ndarray, float, float, float]:
    """Function from DyCheck's repo. Load scene info from json.

    Args:
        data_dir: data path

    Returns:
        A tuple of scene info: center, scale, near, far
    """
    scene_dict = load_from_json(data_dir / "scene.json")
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]
    return center, scale, near, far


def _load_metadata_info(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function from DyCheck's repo. Load scene metadata from json.

    Args:
        data_dir: data path

    Returns:
        A tuple of scene info: frame_names_map, time_ids, camera_ids
    """
    dataset_dict = load_from_json(data_dir / "dataset.json")
    _frame_names = np.array(dataset_dict["ids"])

    metadata_dict = load_from_json(data_dir / "metadata.json")
    time_ids = np.array([metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32)
    camera_ids = np.array([metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32)

    frame_names_map = np.zeros((time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype)
    for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
        frame_names_map[t, c] = _frame_names[i]

    return frame_names_map, time_ids, camera_ids


def _rescale_depth(depth_raw: np.ndarray, cam: Dict) -> np.ndarray:
    """Depth rescale function from DyCheck.

    Args:
        depth: A numpy ndarray of the raw depth
        cam: Dict of the camera

    Returns:
        A numpy ndarray of the processed depth
    """
    xx, yy = np.meshgrid(np.arange(cam["width"], dtype=np.float32), np.arange(cam["height"], dtype=np.float32))
    pixels = np.stack([xx, yy], axis=-1)
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - cam["cy"]) / cam["fy"]
    x = (pixels[..., 0] - cam["cx"]) / cam["fx"]
    # x = (pixels[..., 0] - self.principal_point_x - y * self.skew) / self.scale_factor_x
    # assume skew = 0
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = (cam["camera_to_worlds"][:3, :3] @ local_viewdirs[..., None])[..., 0]
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = viewdirs.reshape((*batch_shape, 3))
    cosa = viewdirs @ (cam["camera_to_worlds"][:, 2])
    depth = depth_raw / cosa[..., None]
    return depth


@dataclass
class DycheckDataParserConfig(DataParserConfig):
    """Dycheck (https://arxiv.org/abs/2210.13445) dataset parser config"""

    _target: Type = field(default_factory=lambda: Dycheck)
    """target class to instantiate"""
    data: Path = Path("data/iphone/mochi-high-five")
    """Directory specifying location of data."""
    scale_factor: float = 5.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    downscale_factor: int = 1
    """How much to downscale images."""
    scene_box_bound: float = 1.5
    """Boundary of scene box."""


@dataclass
class Dycheck(DataParser):
    """Dycheck (https://arxiv.org/abs/2210.13445) Dataset `iphone` subset"""

    config: DycheckDataParserConfig
    includes_time: bool = True

    def __init__(self, config: DycheckDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        # load extra info from "extra.json"
        extra_path = self.data / "extra.json"
        extra_dict = load_from_json(extra_path)
        self._factor = extra_dict["factor"]
        self._fps = extra_dict["fps"]
        self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
        self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
        self._up = np.array(extra_dict["up"], dtype=np.float32)
        self._center, self._scale, self._near, self._far = _load_scene_info(self.data)
        self._frame_names_map, self._time_ids, self._camera_ids = _load_metadata_info(self.data)

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        splits_dir = self.data / "splits"

        # scale the scene to fill the aabb bbox
        sf = self.config.scene_box_bound / 4 / (self._scale * self._far)
        # CONSOLE.print(f"scale factor changed from {self.config.scale_factor} to {sf}")
        self.config.scale_factor = sf

        if not (splits_dir / f"{split}.json").exists():
            CONSOLE.print(f"split {split} not found, using split train")
            split = "train"
        split_dict = load_from_json(splits_dir / f"{split}.json")
        frame_names = np.array(split_dict["frame_names"])
        time_ids = np.array(split_dict["time_ids"])
        if split != "train":
            CONSOLE.print(f"split {split} is empty, using the 1st training image")
            split_dict = load_from_json(splits_dir / "train.json")
            frame_names = np.array(split_dict["frame_names"])[[0]]
            time_ids = np.array(split_dict["time_ids"])[[0]]

        image_filenames, depth_filenames, cams = self.process_frames(frame_names, time_ids)

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.config.scene_box_bound] * 3, [self.config.scene_box_bound] * 3], dtype=torch.float32
            )
        )
        cam_dict = {}
        for k in cams[0].keys():
            cam_dict[k] = torch.stack([torch.as_tensor(c[k]) for c in cams], dim=0)
        cameras = Cameras(camera_type=CameraType.PERSPECTIVE, **cam_dict)

        scale = self._scale * self.config.scale_factor
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            metadata={
                "depth_filenames": depth_filenames,
                "depth_unit_scale_factor": scale,
                "scale": scale,
                "near": self._near * scale,
                "far": self._far * scale,
            },
        )

        return dataparser_outputs

    def process_frames(self, frame_names: List[str], time_ids: np.ndarray) -> Tuple[List, List, List]:
        """Read cameras and filenames from the name list.

        Args:
            frame_names: list of file names.
            time_ids: time id of each frame.

        Returns:
            A list of camera, each entry is a dict of the camera.
        """
        image_filenames, depth_filenames = [], []
        cams = []
        for idx, frame in enumerate(frame_names):
            image_filenames.append(self.data / f"rgb/{self.config.downscale_factor}x/{frame}.png")
            depth_filenames.append(self.data / f"processed_depth/{self.config.downscale_factor}x/{frame}.npy")
            cam_json = load_from_json(self.data / f"camera/{frame}.json")
            c2w = torch.as_tensor(cam_json["orientation"]).T
            position = torch.as_tensor(cam_json["position"])
            position -= self._center  # some scenes look weird (wheel)
            position *= self._scale * self.config.scale_factor
            pose = torch.zeros([3, 4])
            pose[:3, :3] = c2w
            pose[:3, 3] = position
            # from opencv coord to opengl coord (used by nerfstudio)
            pose[0:3, 1:3] *= -1  # switch cam coord x,y
            pose = pose[[1, 0, 2], :]  # switch world x,y
            pose[2, :] *= -1  # invert world z
            # for aabb bbox usage
            pose = pose[[1, 2, 0], :]  # switch world xyz to zxy
            cams.append(
                {
                    "camera_to_worlds": pose,
                    "fx": cam_json["focal_length"] / self.config.downscale_factor,
                    "fy": cam_json["focal_length"] * cam_json["pixel_aspect_ratio"] / self.config.downscale_factor,
                    "cx": cam_json["principal_point"][0] / self.config.downscale_factor,
                    "cy": cam_json["principal_point"][1] / self.config.downscale_factor,
                    "height": cam_json["image_size"][1] // self.config.downscale_factor,
                    "width": cam_json["image_size"][0] // self.config.downscale_factor,
                    "times": torch.as_tensor(time_ids[idx] / self._time_ids.max()).float(),
                }
            )

        d = self.config.downscale_factor
        if not image_filenames[0].exists():
            CONSOLE.print(f"downscale factor {d}x not exist, converting")
            ori_h, ori_w = cv2.imread(str(self.data / f"rgb/1x/{frame_names[0]}.png")).shape[:2]
            (self.data / f"rgb/{d}x").mkdir(exist_ok=True)
            h, w = ori_h // d, ori_w // d
            for frame in frame_names:
                cv2.imwrite(
                    str(self.data / f"rgb/{d}x/{frame}.png"),
                    cv2.resize(cv2.imread(str(self.data / f"rgb/1x/{frame}.png")), [h, w]),
                )
            CONSOLE.print("finished")

        if not depth_filenames[0].exists():
            CONSOLE.print(f"processed depth downscale factor {d}x not exist, converting")
            (self.data / f"processed_depth/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                depth = np.load(self.data / f"depth/1x/{frame}.npy")
                mask = rescale((depth != 0).astype(np.uint8) * 255, 1 / d, cv2.INTER_AREA)
                depth = rescale(depth, 1 / d, cv2.INTER_AREA)
                depth[mask != 255] = 0
                depth = _rescale_depth(depth, cams[idx])
                np.save(str(self.data / f"processed_depth/{d}x/{frame}.npy"), depth)
            CONSOLE.print("finished")

        return image_filenames, depth_filenames, cams
