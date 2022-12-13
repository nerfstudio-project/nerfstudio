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
""" Data parser for pre-prepared datasets with all data stored to disk. """

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class DiskDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: DiskDataParser)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")


@dataclass
class DiskDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: DiskDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        with open(self.config.data / split, mode="r", encoding="utf-8") as f:
            data = np.load(f)

        image_filenames = data["image_filenames"].tolist()
        mask_filenames = data["mask_filenames"].tolist() if "mask_filenames" in data.keys() else None
        metadata = {"semantics": data["semantic_filenames"].tolist()} if "semantic_filenames" in data.keys() else None
        times = data["times"].to_list() if "times" in data.keys() else None

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
        return dataparser_outputs
