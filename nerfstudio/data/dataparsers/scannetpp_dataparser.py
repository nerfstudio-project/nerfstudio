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
"""Data parser for ScanNet++ datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ScanNetppDataParserConfig(DataParserConfig):
    """ScanNet++ dataset config.
    ScanNet++ dataset (https://kaldir.vc.in.tum.de/scannetpp/) is a real-world 3D indoor dataset for semantics understanding and novel view synthesis.
    This dataparser follow the file structure of the dataset.
    Expected structure of the directory:

    .. code-block:: text

        root/
        ├── SCENE_ID0
            ├── dslr
                ├── resized_images
                ├── resized_anon_masks
                ├── nerfstudio/transforms.json
        ├── SCENE_ID1/
        ...
    """

    _target: Type = field(default_factory=lambda: ScanNetpp)
    """target class to instantiate"""
    data: Path = Path("scannetpp/410c470782")
    """Directory to the root of the data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.5
    """How much to scale the region of interest by. Default is 1.5 since the cameras are inside the rooms."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    images_dir: Path = Path("dslr/resized_images")
    """Relative path to the images directory (default: resized_images)"""
    masks_dir: Path = Path("dslr/resized_anon_masks")
    """Relative path to the masks directory (default: resized_anon_masks)"""
    transforms_path: Path = Path("dslr/nerfstudio/transforms.json")
    """Relative path to the transforms.json file"""


@dataclass
class ScanNetpp(DataParser):
    """ScanNet++ DatasetParser"""

    config: ScanNetppDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        meta = load_from_json(self.config.data / self.config.transforms_path)
        data_dir = self.config.data / self.config.images_dir
        mask_dir = self.config.data / self.config.masks_dir

        image_filenames = []
        mask_filenames = []
        poses = []
        i_train = []
        i_eval = []
        # sort the frames by fname
        frames = meta["frames"] + meta["test_frames"]
        test_frames = [f["file_path"] for f in meta["test_frames"]]
        frames.sort(key=lambda x: x["file_path"])

        for idx, frame in enumerate(frames):
            filepath = Path(frame["file_path"])
            fname = data_dir / filepath

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if meta.get("has_mask", True) and "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = mask_dir / mask_filepath
                mask_filenames.append(mask_fname)

            if frame["file_path"] in test_frames:
                i_eval.append(idx)
            else:
                i_train.append(idx)

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        if not self.config.auto_scale_poses:
            # Set aabb_scale to scene_scale * the max of the absolute values of the poses
            aabb_scale = self.config.scene_scale * float(torch.max(torch.abs(poses[:, :3, 3])))
        else:
            aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])
        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={},
        )
        return dataparser_outputs
