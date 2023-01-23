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

"""Data parser for instant ngp data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Type

import imageio
import numpy as np
import torch
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)


@dataclass
class InstantNGPDataParserConfig(DataParserConfig):
    """Instant-NGP dataset parser config"""

    _target: Type = field(default_factory=lambda: InstantNGP)
    """target class to instantiate"""
    data: Path = Path("data/ours/posterv2")
    """Directory or explicit json file path specifying location of data."""
    scene_scale: float = 0.3333
    """How much to scale the scene."""


@dataclass
class InstantNGP(DataParser):
    """Instant NGP Dataset"""

    config: InstantNGPDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        for frame in meta["frames"]:
            fname = data_dir / Path(frame["file_path"])
            # search for png file
            if not fname.exists():
                fname = data_dir / Path(frame["file_path"] + ".png")
            if not fname.exists():
                CONSOLE.log(f"couldnt find {fname} image")
                num_skipped_image_filenames += 1
            else:
                if "w" not in meta:
                    img_0 = imageio.imread(fname)
                    h, w = img_0.shape[:2]
                    meta["w"] = w
                    if "h" in meta:
                        meta_h = meta["h"]
                        assert meta_h == h, f"height of image dont not correspond metadata {h} != {meta_h}"
                    else:
                        meta["h"] = h
                image_filenames.append(fname)
                poses.append(np.array(frame["transform_matrix"]))
        if num_skipped_image_filenames >= 0:
            CONSOLE.print(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        poses = np.array(poses).astype(np.float32)
        poses[:, :3, 3] *= self.config.scene_scale

        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta.get("k1", 0)),
            k2=float(meta.get("k2", 0)),
            k3=float(meta.get("k3", 0)),
            k4=float(meta.get("k4", 0)),
            p1=float(meta.get("p1", 0)),
            p2=float(meta.get("p2", 0)),
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 0.5 * meta.get("aabb_scale", 1)

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fl_x, fl_y = InstantNGP.get_focal_lengths(meta)

        w, h = meta["w"], meta["h"]

        cameras = Cameras(
            fx=float(fl_x),
            fy=float(fl_y),
            cx=float(meta.get("cx", 0.5 * w)),
            cy=float(meta.get("cy", 0.5 * h)),
            distortion_params=distortion_params,
            height=int(h),
            width=int(w),
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO(ethan): add alpha background color
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.config.scene_scale,
        )

        return dataparser_outputs

    @classmethod
    def get_focal_lengths(cls, meta: Dict) -> Tuple[float, float]:
        """Reads or computes the focal length from transforms dict.
        Args:
            meta: metadata from transforms.json file.
        Returns:
            Focal lengths in the x and y directions. Error is raised if these cannot be calculated.
        """
        fl_x, fl_y = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / np.tan(0.5 * rad)

        if "fl_x" in meta:
            fl_x = meta["fl_x"]
        elif "x_fov" in meta:
            fl_x = fov_to_focal_length(np.deg2rad(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fl_x = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "camera_angle_y" not in meta or "y_fov" not in meta:
            fl_y = fl_x
        else:
            if "fl_y" in meta:
                fl_y = meta["fl_y"]
            elif "y_fov" in meta:
                fl_y = fov_to_focal_length(np.deg2rad(meta["y_fov"]), meta["h"])
            elif "camera_angle_y" in meta:
                fl_y = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fl_x == 0 or fl_y == 0:
            raise AttributeError("Focal length cannot be calculated from transforms.json (missing fields).")

        return (fl_x, fl_y)
