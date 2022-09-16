# Copyright 2022 The Plenoptix Team. All rights reserved.
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

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nerfactory.cameras import utils as camera_utils
from nerfactory.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfactory.configs import base as cfg
from nerfactory.datamanagers.dataparsers.base import DataParser
from nerfactory.datamanagers.structs import DatasetInputs, SceneBounds
from nerfactory.utils.io import get_absolute_path, load_from_json


@dataclass
class Nerfactory(DataParser):
    """Nerfactory Dataset"""

    config: cfg.NerfactoryDataParserConfig

    def _generate_dataset_inputs(self, split="train"):

        abs_dir = get_absolute_path(self.config.data_directory)

        meta = load_from_json(abs_dir / "transforms.json")
        image_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        for frame in meta["frames"]:
            fname = abs_dir / Path(frame["file_path"])
            if self.config.downscale_factor > 1:
                fname = abs_dir / f"images_{self.config.downscale_factor}" / Path(frame["file_path"]).name
            else:
                fname = abs_dir / Path(frame["file_path"])
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
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses = camera_utils.auto_orient_poses(poses, method=self.config.orientation_method)

        # Scale poses
        scale_factor = 1.0 / torch.max(torch.abs(poses[:, :3, 3]))
        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_bounds = SceneBounds(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )

        cameras = Cameras(
            fx=float(meta["fl_x"]),
            fy=float(meta["fl_y"]),
            cx=float(meta["cx"]),
            cy=float(meta["cy"]),
            distortion_params=distortion_params,
            height=int(meta["h"]),
            width=int(meta["w"]),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_bounds=scene_bounds,
        )
        return dataset_inputs
