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
""" Data parser for pre-prepared datasets for all cameras, with no additional processing needed"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics
)
from nerfstudio.data.scene_box import SceneBox


@dataclass
class MinimalDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: MinimalDataParser)
    """target class to instantiate"""
    data: Path = Path("/home/nikhil/nerfstudio-main/tests/data/lego_test/minimal_parser")


@dataclass
class MinimalDataParser(DataParser):
    """Nerfstudio DatasetParser"""

    config: MinimalDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements
        filepath = self.config.data / f"{split}.npz"
        data = np.load(filepath, allow_pickle=True)

        image_filenames = data["image_filenames"].tolist()
        mask_filenames = data["mask_filenames"].tolist() if "mask_filenames" in data.keys() else None
        
        metadata = None
        if "semantics" in data.keys():
            semantics = data["semantics"].item()
            metadata = {
                "semantics": Semantics(
                    filenames=semantics["filenames"].tolist(), 
                    classes=semantics["classes"].tolist(), 
                    colors=semantics["colors"].tolist(), 
                    mask_classes=semantics["mask_classes"].tolist()
                )
            }

        scene_box_aabb = torch.from_numpy(data["scene_box"])
        scene_box = SceneBox(aabb=scene_box_aabb)

        camera_np = data["cameras"].item()
        cameras = Cameras(
            fx=torch.from_numpy(camera_np["fx"]),
            fy=torch.from_numpy(camera_np["fy"]),
            cx=torch.from_numpy(camera_np["cx"]),
            cy=torch.from_numpy(camera_np["cy"]),
            distortion_params=torch.from_numpy(camera_np["distortion_params"]) if "distortion_params" in camera_np.keys() else None ,
            height=torch.from_numpy(camera_np["height"]),
            width=torch.from_numpy(camera_np["width"]),
            camera_to_worlds=torch.from_numpy(camera_np["camera_to_worlds"])[:, :3, :4],
            camera_type=torch.from_numpy(camera_np["camera_type"]),
            times=torch.from_numpy(camera_np["times"]) if "times" in camera_np.keys() else None,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            metadata=metadata,
        )
        return dataparser_outputs
