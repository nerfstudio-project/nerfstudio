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

"""Data parser for friends dataset"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nerfactory.cameras.cameras import Cameras, CameraType
from nerfactory.datamanagers.dataparsers.base import DataParser
from nerfactory.datamanagers.datasets import InputDataset
from nerfactory.datamanagers.structs import (
    DatasetInputs,
    PointCloud,
    SceneBounds,
    Semantics,
)
from nerfactory.utils.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from nerfactory.utils.io import get_absolute_path, load_from_json


def get_semantics_and_masks(self: InputDataset, image_idx: int):
    """function to process additional semantics and mask information"""
    # handle mask
    person_index = self.inputs.additional_inputs["semantics"].thing_classes.index("person")
    thing_image_filename = self.inputs.additional_inputs["semantics"].thing_filenames[image_idx]
    pil_image = Image.open(thing_image_filename)
    thing_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
    mask = (thing_semantics != person_index).to(torch.float32)  # 1 where valid
    # handle semantics
    stuff_image_filename = self.inputs.additional_inputs["semantics"].stuff_filenames[image_idx]
    pil_image = Image.open(stuff_image_filename)
    stuff_semantics = torch.from_numpy(np.array(pil_image, dtype="int32"))[..., None]
    return {"mask": mask, "semantics": stuff_semantics}


@dataclass
class Friends(DataParser):
    """Friends Dataset

    Args:
        data_directory: Location of data
        include_semantics: whether or not to include the semantics. Defaults to False.
        include_point_cloud: whether or not to include the point cloud. Defaults to False.
    """

    data_directory: Path
    include_semantics: bool = True
    include_point_cloud: bool = False

    @classmethod
    def _get_aabb_and_transform(cls, basedir):
        """Returns the aabb and pointcloud transform from the threejs.json file."""
        filename = basedir / "threejs.json"
        assert filename.exists()
        data = load_from_json(filename)

        # point cloud transformation
        transposed_point_cloud_transform = np.array(data["object"]["children"][0]["matrix"]).reshape(4, 4).T
        assert transposed_point_cloud_transform[3, 3] == 1.0

        # bbox transformation
        bbox_transform = np.array(data["object"]["children"][1]["matrix"]).reshape(4, 4).T
        w, h, d = data["geometries"][1]["width"], data["geometries"][1]["height"], data["geometries"][1]["depth"]
        temp = np.array([w, h, d]) / 2.0
        bbox = np.array([-temp, temp])
        bbox = np.concatenate([bbox, np.ones_like(bbox[:, 0:1])], axis=1)
        bbox = (bbox_transform @ bbox.T).T[:, 0:3]

        aabb = bbox  # rename to aabb because it's an axis-aligned bounding box
        return torch.from_numpy(aabb).float(), torch.from_numpy(transposed_point_cloud_transform).float()

    def _generate_dataset_inputs(self, split="train"):  # pylint: disable=too-many-statements

        abs_dir = get_absolute_path(self.data_directory)

        images_data = read_images_binary(abs_dir / "colmap" / "images.bin")
        # `image_path` is only the end of the filename, including the extension e.g., `.jpg`
        image_paths = sorted((abs_dir / "images").iterdir())

        image_path_to_image_id = {}
        image_id_to_image_path = {}
        for v in images_data.values():
            image_path_to_image_id[v.name] = v.id
            image_id_to_image_path[v.id] = v.name
        # TODO: handle the splits differently
        image_filenames = [abs_dir / "images" / image_path for image_path in image_paths]

        # -- set the bounding box ---
        aabb, transposed_point_cloud_transform = self._get_aabb_and_transform(abs_dir)
        scene_bounds_original = SceneBounds(aabb=aabb)
        # for shifting and rescale accoding to scene bounds
        box_center = scene_bounds_original.get_center()
        box_scale_factor = 5.0 / scene_bounds_original.get_diagonal_length()  # the target diagonal length
        scene_bounds = scene_bounds_original.get_centered_and_scaled_scene_bounds(box_scale_factor)

        # --- intrinsics ---
        cameras_data = read_cameras_binary(abs_dir / "colmap" / "cameras.bin")
        focal_lengths = []
        for image_path in image_paths:
            cam = cameras_data[image_path_to_image_id[image_path]]
            assert len(cam.params) == 3
            focal_lengths.append(cam.params[0])  # f (fx and fy)
        focal_lengths = torch.stack(focal_lengths, dim=0)

        cam = cameras_data[image_path_to_image_id[image_paths[0]]]
        cx = cam.params[1]  # cx
        cy = cam.params[2]  # cy

        # --- camera_to_world (extrinsics) ---
        camera_to_world = []
        bottom_row = np.array([0, 0, 0, 1.0]).reshape(1, 4)
        for image_path in image_paths:
            image_data = images_data[image_path_to_image_id[image_path]]
            rot = image_data.qvec2rotmat()
            trans = image_data.tvec.reshape(3, 1)
            c2w = np.concatenate([np.concatenate([rot, trans], 1), bottom_row], 0)
            camera_to_world.append(c2w)
        camera_to_world = torch.tensor(np.array(camera_to_world)).float()
        camera_to_world = torch.inverse(camera_to_world)
        camera_to_world[..., 1:3] *= -1
        camera_to_world = transposed_point_cloud_transform @ camera_to_world
        camera_to_world = camera_to_world[:, :3]
        camera_to_world[..., 3] = (camera_to_world[..., 3] - box_center) * box_scale_factor  # center and rescale

        # --- semantics ---
        semantics = None
        if self.include_semantics:
            thing_filenames = [
                Path(str(image_filename).replace("/images/", "/segmentations/thing/").replace(".jpg", ".png"))
                for image_filename in image_filenames
            ]
            stuff_filenames = [
                Path(str(image_filename).replace("/images/", "/segmentations/stuff/").replace(".jpg", ".png"))
                for image_filename in image_filenames
            ]
            panoptic_classes = load_from_json(abs_dir / "panoptic_classes.json")
            stuff_classes = panoptic_classes["stuff"]
            stuff_colors = torch.tensor(panoptic_classes["stuff_colors"], dtype=torch.float32) / 255.0
            thing_classes = panoptic_classes["thing"]
            thing_colors = torch.tensor(panoptic_classes["thing_colors"], dtype=torch.float32) / 255.0
            semantics = Semantics(
                stuff_classes=stuff_classes,
                stuff_colors=stuff_colors,
                stuff_filenames=stuff_filenames,
                thing_classes=thing_classes,
                thing_colors=thing_colors,
                thing_filenames=thing_filenames,
            )

        # Possibly include the sparse point cloud from COLMAP in the dataset inputs.
        # NOTE(ethan): this will be common across the different splits.
        point_cloud = PointCloud()
        if self.include_point_cloud:
            points_3d = read_points3d_binary(abs_dir / "colmap" / "points3D.bin")
            xyz = torch.tensor(np.array([p_value.xyz for p_id, p_value in points_3d.items()])).float()
            rgb = torch.tensor(np.array([p_value.rgb for p_id, p_value in points_3d.items()])).float()
            xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], -1)
            xyz = (xyz_h @ transposed_point_cloud_transform.T)[..., :3]
            xyz = (xyz - box_center) * box_scale_factor  # center and rescale
            point_cloud.xyz = xyz
            point_cloud.rgb = rgb

        cameras = Cameras(
            fx=focal_lengths,
            fy=focal_lengths,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            cameras=cameras,
            point_cloud=point_cloud,
            scene_bounds=scene_bounds,
            additional_inputs={"semantics": {"data": semantics, "func": get_semantics_and_masks}},
        )
        return dataset_inputs
