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
"""
Data parser for pre-prepared datasets for all cameras, with no additional processing needed
Optional fields - semantics, mask_filenames, cameras.distortion_params, cameras.times
"""

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
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox


@dataclass
class MinimalDataParserConfig(DataParserConfig):
    """Minimal dataset config"""

    _target: Type = field(default_factory=lambda: MinimalDataParser)
    """target class to instantiate"""
    data: Path = Path("/home/nikhil/nerfstudio-main/tests/data/lego_test/minimal_parser")


@dataclass
class MinimalDataParser(DataParser):
    """Minimal DatasetParser"""

    config: MinimalDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        filepath = self.config.data / f"{split}.npz"
        data = np.load(filepath, allow_pickle=True)

        image_filenames = [filepath.parent / path for path in data["image_filenames"].tolist()]
        mask_filenames = None
        if "mask_filenames" in data.keys():
            mask_filenames = [filepath.parent / path for path in data["mask_filenames"].tolist()]

        metadata = None
        if "semantics" in data.keys():
            semantics = data["semantics"].item()
            metadata = {
                "semantics": Semantics(
                    filenames=[filepath.parent / path for path in semantics["filenames"].tolist()],
                    classes=semantics["classes"].tolist(),
                    colors=torch.from_numpy(semantics["colors"]),
                    mask_classes=semantics["mask_classes"].tolist(),
                )
            }

        scene_box_aabb = torch.from_numpy(data["scene_box"])
        scene_box = SceneBox(aabb=scene_box_aabb)

        camera_np = data["cameras"].item()
        distortion_params = None
        if "distortion_params" in camera_np.keys():
            distortion_params = torch.from_numpy(camera_np["distortion_params"])
        cameras = Cameras(
            fx=torch.from_numpy(camera_np["fx"]),
            fy=torch.from_numpy(camera_np["fy"]),
            cx=torch.from_numpy(camera_np["cx"]),
            cy=torch.from_numpy(camera_np["cy"]),
            distortion_params=distortion_params,
            height=torch.from_numpy(camera_np["height"]),
            width=torch.from_numpy(camera_np["width"]),
            camera_to_worlds=torch.from_numpy(camera_np["camera_to_worlds"])[:, :3, :4],
            camera_type=torch.from_numpy(camera_np["camera_type"]),
            times=torch.from_numpy(camera_np["times"]) if "times" in camera_np.keys() else None,
        )

        applied_scale = 1.0
        applied_transform = np.eye(4, dtype=np.float32)[:3, :]
        if "applied_scale" in data.keys():
            applied_scale = float(data["applied_scale"])
        if "applied_transform" in data.keys():
            applied_transform = data["applied_transform"].astype(np.float32)
            assert applied_transform.shape == (3, 4)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            dataparser_transform=applied_transform,
            dataparser_scale=applied_scale,
            metadata=metadata,
        )
        return dataparser_outputs
