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

"""Phototourism dataset parser."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.utils.colmap_utils import read_cameras_binary, read_images_binary


@dataclass
class PhototourismDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: Phototourism)
    """target class to instantiate"""
    data: Path = Path("data/trevi_fountain")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    split_frac: float = 0.8
    """fraction of data to use for training"""


@dataclass
class Phototourism(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: PhototourismDataParserConfig

    def __init__(self, config: PhototourismDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.split_frac = config.split_frac

    def _generate_dataparser_outputs(self, split="train"):

        cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
        imgs = read_images_binary(self.data / "dense/sparse/images.bin")

        cs = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        image_filenames = []

        for _id, cam in cams.items():
            img = imgs[_id]

            assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

            cs.append(torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1))
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))

            image_filenames.append(self.data / "dense/images" / img.name)

        cs = torch.stack(cs)
        fxs = torch.stack(fxs)
        fys = torch.stack(fys)
        cxs = torch.stack(cxs)
        cys = torch.stack(cys)

        cameras = Cameras(
            camera_to_worlds=cs,
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            camera_type=CameraType.PERSPECTIVE,
        )

        train_idx = int(len(image_filenames) * self.split_frac)

        if split == "train":
            cameras = cameras[:train_idx]
            image_filenames = image_filenames[:train_idx]
        else:
            cameras = cameras[train_idx:]
            image_filenames = image_filenames[train_idx:]

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
        )

        return dataparser_outputs
