# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

""" 3D Gaussian Splatting data parser """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import torch
from jaxtyping import Float
from pykdtree.kdtree import KDTree

from nerfstudio.cameras.points import Gaussians3D, Points3D

from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
)
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.utils.rich_utils import CONSOLE


def inverse_sigmoid(x):
    # TODO: move this to some utils file
    return torch.log(1 / (x - 1))


@dataclass
class GaussianSplattingDataParserConfig(ColmapDataParserConfig):
    """Gaussian splatting dataset config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingDataParser)
    """Target class to initiate"""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    initial_opacity_value: Float = 0.1
    """Initial opacity value (original paper has 0.1)"""
    intial_scale: Literal["mean_3_nn", "rand", "ones"] = "mean_3_nn"
    """Method to initialize gaussian scales"""


class GaussianSplattingDataParser(ColmapDataParser):
    """Gaussian splatting dataparser

    Similar to colmap datapareser, but adds support for SFM points.
    """

    config: GaussianSplattingDataParserConfig

    def __init__(self, config: GaussianSplattingDataParserConfig):
        super().__init__(config=config)  # type: ignore
        self.config = config

    def _get_all_sfm_points(self) -> Points3D:
        """get initial SfM points"""
        recon_dir = self.config.data / self.config.colmap_path
        if (recon_dir / "cameras.txt").exists():
            points = colmap_utils.read_points3D_text(recon_dir / "points3D.txt")
        elif (recon_dir / "cameras.bin").exists():
            points = colmap_utils.read_points3D_binary(recon_dir / "points3D.bin")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {recon_dir}")

        points_ids = []
        points_xyzs = []
        points_rgbs = []
        points_errors = []
        for idx, data in points.items():
            points_ids.append(torch.Tensor([data.id]))
            points_xyzs.append(torch.from_numpy(data.xyz))
            points_rgbs.append(torch.from_numpy(data.rgb))
            points_errors.append(torch.from_numpy(data.error))

        points_ids = torch.stack(points_ids)
        points_xyzs = torch.stack(points_xyzs)
        points_rgbs = torch.stack(points_rgbs)
        points_errors = torch.stack(points_errors)

        points = Points3D(
            ids=points_ids,
            xyzs=points_xyzs,
            rgbs=points_rgbs,
            errors=points_errors,
        )
        return points

    def get_initial_gaussians(self) -> Gaussians3D:
        """Initialize 3D gaussians from SfM points"""
        points: Points3D = self._get_all_sfm_points()
        num_points = points.xyzs.shape[0]
        assert num_points > 0, "Not enough SfM points to initialize Gaussians"
        CONSOLE.log("Number of Gaussians at initialisation : ", len(points.xyzs))

        xyzs = points.xyzs.cpu().numpy()
        opacity = inverse_sigmoid(0.1 * torch.ones((num_points, 1), dtype=torch.float, device="cuda"))

        if self.config.intial_scale == "mean_3_nn":
            kd_tree = KDTree(xyzs)
            dist, idx = kd_tree.query(xyzs, k=4)
            mean_min_three_dis = dist[:, 1:].mean(axis=1)
            mean_min_three_dis = torch.Tensor(mean_min_three_dis).to(torch.float32)  # * scale_init_value
            scale = torch.ones(num_points, 3).to(torch.float32) * mean_min_three_dis.unsqueeze(dim=1)
        elif self.config.intial_scale == "rand":
            scale = torch.rand(size=(num_points, 3)).to(torch.float32)
        elif self.config.intial_scale == "ones":
            scale = torch.ones(size=(num_points, 3)).to(torch.float32)

        quat = torch.Tensor([1, 0, 0, 0]).unsqueeze(dim=0).repeat(num_points, 1).to(torch.float32)

        gaussians = Gaussians3D(
            positions=points.xyzs,
            rgbs=points.rgbs,
            opacity=opacity,
            quat=quat,
            scale=scale,
        )

        return gaussians
