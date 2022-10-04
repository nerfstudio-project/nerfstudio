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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class InstantNGPDataParserConfig(DataParserConfig):
    """Instant-NGP dataset parser config"""

    _target: Type = field(default_factory=lambda: InstantNGP)
    """target class to instantiate"""
    data: Path = Path("data/ours/posterv2")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 0.33
    """How much to scale the scene."""


@dataclass
class InstantNGP(DataParser):
    """Instant NGP Dataset"""

    config: InstantNGPDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        for frame in meta["frames"]:
            fname = self.config.data / Path(frame["file_path"])
            if not fname:
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
        poses[:3, 3] *= self.config.scene_scale

        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]), k2=float(meta["k2"]), p1=float(meta["p1"]), p2=float(meta["p2"])
        )

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = meta["aabb_scale"]
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=float(meta["fl_x"]),
            fy=float(meta["fl_y"]),
            cx=float(meta["cx"]),
            cy=float(meta["cy"]),
            distortion_params=distortion_params,
            height=int(meta["h"]),
            width=int(meta["w"]),
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO(ethan): add alpha background color
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )

        return dataparser_outputs
