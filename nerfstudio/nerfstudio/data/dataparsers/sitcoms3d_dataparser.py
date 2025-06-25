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

"""Data parser for sitcoms3D dataset.

The dataset is from the paper ["The One Where They Reconstructed 3D Humans and
Environments in TV Shows"](https://ethanweber.me/sitcoms3D/)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class Sitcoms3DDataParserConfig(DataParserConfig):
    """sitcoms3D dataset parser config"""

    _target: Type = field(default_factory=lambda: Sitcoms3D)
    """target class to instantiate"""
    data: Path = Path("data/sitcoms3d/TBBT-big_living_room")
    """Directory specifying location of data."""
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    downscale_factor: int = 4
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Sitcoms3D axis-aligned bbox will be scaled to this value.
    """


@dataclass
class Sitcoms3D(DataParser):
    """Sitcoms3D Dataset"""

    config: Sitcoms3DDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        cameras_json = load_from_json(self.config.data / "cameras.json")
        frames = cameras_json["frames"]
        bbox = torch.tensor(cameras_json["bbox"])

        downscale_suffix = f"_{self.config.downscale_factor}" if self.config.downscale_factor != 1 else ""
        images_folder = f"images{downscale_suffix}"
        segmentations_folder = f"segmentations{downscale_suffix}"

        image_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for frame in frames:
            # unpack data
            image_filename = self.config.data / images_folder / frame["image_name"]
            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])[:3]
            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # rotate the cameras and box 90 degrees about the x axis to put the z axis up
        rotation = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        camera_to_worlds[:, :3] = rotation @ camera_to_worlds[:, :3]
        bbox = (rotation @ bbox.T).T

        scene_scale = self.config.scene_scale

        # -- set the scene box ---
        scene_box = SceneBox(aabb=bbox)
        # center the box and adjust the cameras too
        center = scene_box.get_center()
        scene_box.aabb -= center
        camera_to_worlds[..., 3] -= center
        # scale the longest dimension to match the cube size
        lengths = scene_box.aabb[1] - scene_box.aabb[0]
        longest_dim = torch.argmax(lengths)
        longest_length = lengths[longest_dim]
        scale = scene_scale / longest_length.item()
        scene_box.aabb = scene_box.aabb * scale  # box
        camera_to_worlds[..., 3] *= scale  # cameras

        # --- semantics ---
        semantics = None
        if self.config.include_semantics:
            empty_path = Path()
            replace_this_path = str(empty_path / images_folder / empty_path)
            with_this_path = str(empty_path / segmentations_folder / "thing" / empty_path)
            filenames = [
                Path(str(image_filename).replace(replace_this_path, with_this_path).replace(".jpg", ".png"))
                for image_filename in image_filenames
            ]
            panoptic_classes = load_from_json(self.config.data / "panoptic_classes.json")
            classes = panoptic_classes["thing"]
            colors = torch.tensor(panoptic_classes["thing_colors"], dtype=torch.float32) / 255.0
            semantics = Semantics(filenames=filenames, classes=classes, colors=colors, mask_classes=["person"])

        assert torch.all(cx[0] == cx), "Not all cameras have the same cx. Our Cameras class does not support this."
        assert torch.all(cy[0] == cy), "Not all cameras have the same cy. Our Cameras class does not support this."

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=float(cx[0]),
            cy=float(cy[0]),
            camera_to_worlds=camera_to_worlds,
            camera_type=CameraType.PERSPECTIVE,
        )
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata={"semantics": semantics} if self.config.include_semantics else {},
            dataparser_scale=scale,
        )
        return dataparser_outputs
