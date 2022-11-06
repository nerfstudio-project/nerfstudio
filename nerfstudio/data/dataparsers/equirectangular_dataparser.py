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

"""Data parser for record3d dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
from rich.console import Console
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils import poses as pose_utils
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)


@dataclass
class EquirectangularDataParserConfig(DataParserConfig):
    """Equirectangular dataset config"""

    _target: Type = field(default_factory=lambda: Equirectangular)
    """target class to instantiate"""
    data: Path = Path("data/record3d/bear")
    """Location of data"""
    val_skip: int = 8
    """1/val_skip images to use for validation."""
    aabb_scale: float = 4.0
    """Scene scale."""
    orientation_method: Literal["pca", "up"] = "up"
    """The method to use for orientation"""


@dataclass
class Equirectangular(DataParser):
    """Equirectangular Dataset"""

    config: EquirectangularDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:

        frames_root = self.config.data / "frames"
        image_filenames = []
        for f in frames_root.iterdir():
            image_filenames.append(f)
        
        poses_arr = np.load(self.config.data / 'poses_bounds.npy')
        intrinsic_arr = np.load(self.config.data / 'hwf_cxcy.npy')

        poses = poses_arr[:, :-2].reshape([-1, 3, 4]).transpose([1,2,0])
        #poses [3, 4, images] --> [images, 3, 4]
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)

        if len(intrinsic_arr) == 3:
            H, W, f = intrinsic_arr
            cx = W / 2.0
            cy =  H / 2.0
            fx = f
            fy = f
        else:
            H, W, fx, fy, cx, cy = intrinsic_arr

        # convert to Tensors
        poses = torch.from_numpy(poses[:, :3, :4])

        poses = camera_utils.auto_orient_and_center_poses(
            pose_utils.to4x4(poses), method=self.config.orientation_method
        )[:, :3, :4]

        # Centering poses
        poses[:, :3, 3] = poses[:, :3, 3] - torch.mean(poses[:, :3, 3], dim=0)
        poses = pose_utils.normalize(poses)

        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32) * self.config.aabb_scale
        scene_box = SceneBox(aabb=aabb)

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_worlds=poses,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )

        return dataparser_outputs
