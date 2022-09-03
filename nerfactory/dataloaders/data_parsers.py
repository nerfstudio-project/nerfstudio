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

"""A set of standard datasets."""

from __future__ import annotations

import dataclasses
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from nerfactory.cameras import utils as camera_utils
from nerfactory.cameras.cameras import Cameras, CameraType
from nerfactory.configs import base as cfg
from nerfactory.dataloaders.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_pointsTD_binary,
)
from nerfactory.dataloaders.structs import (
    DatasetInputs,
    PointCloud,
    SceneBounds,
    Semantics,
)
from nerfactory.utils.colors import get_color
from nerfactory.utils.io import (
    get_absolute_path,
    load_from_json,
    load_from_pkl,
    write_to_pkl,
)
from nerfactory.utils.misc import get_hash_str_from_dict


@dataclass
class DataParser:
    """A dataset."""

    def __init__(self, config: cfg.DataParserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def _generate_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate.

        Returns:
            DatasetInputs
        """

    def get_dataset_inputs(self, split: str = "train", use_preprocessing_cache: bool = False) -> DatasetInputs:
        """Returns the dataset inputs for the given split.

        Args:
            split: Which dataset split to generate.
            use_preprocessing_cache: Whether to use the cached dataset inputs. Defaults to False.

        Returns:
            DatasetInputs
        """
        if use_preprocessing_cache:
            dataset_inputs = self._get_dataset_inputs_from_cache(split)
        else:
            dataset_inputs = self._generate_dataset_inputs(split)
        return dataset_inputs

    def _get_cache_filename(self, split: str) -> Path:
        """Creates a cache filename from the dataset inputs arguments.

        Args:
            split: Which dataset split to generate filename for.

        Returns:
            filename for cache.
        """
        dataset_config_hash = get_hash_str_from_dict(dataclasses.asdict(self))
        dataset_config_hash_filename = get_absolute_path(
            f"cache/dataset_inputs/{dataset_config_hash}-{split}.pkl"
        ).parent.mkdir()
        assert dataset_config_hash_filename is not None, "dataset hash is None"
        return dataset_config_hash_filename

    def save_dataset_inputs_to_cache(self, split: str):
        """Saves the dataset inputs to cache.

        Args:
            split: Which dataset split to save.
        """
        dataset_inputs = self.get_dataset_inputs(split=split)
        dataset_config_hash_filename = self._get_cache_filename(split)
        write_to_pkl(dataset_config_hash_filename, dataset_inputs)

    def _get_dataset_inputs_from_cache(self, split: str) -> DatasetInputs:
        """Loads the dataset inputs from cache. If the cache does not exist, it will be created.

        Args:
            split: Which dataset split to load.
        """
        dataset_config_hash_filename = self._get_cache_filename(split)
        if dataset_config_hash_filename.exists():
            logging.info("Loading dataset from cache.")
            dataset_inputs = load_from_pkl(dataset_config_hash_filename)
        else:
            logging.info("Cache file not found. Generating and saving dataset to cache.")
            dataset_inputs = self._generate_dataset_inputs(split=split)
            self.save_dataset_inputs_to_cache(split)
        return dataset_inputs


@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.

    Args:
        data_directory: Location of data
        alpha_color: Sets transparent regions to specified color, otherwise black.
        scale_factor: How much to scale the camera origins by.
        downscale_factor: How much to downscale images. Defaults to 1.
    """

    def __init__(self, config: cfg.BlenderDataParserConfig):
        super().__init__(config=config)
        self.data_directory: Path = config.data_directory
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.downscale_factor: int = config.downscale_factor

    def _generate_dataset_inputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        abs_dir = get_absolute_path(self.data_directory)
        meta = load_from_json(abs_dir / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = abs_dir / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_bounds = SceneBounds(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            cameras=cameras,
            downscale_factor=self.downscale_factor,
            alpha_color=alpha_color_tensor,
            scene_bounds=scene_bounds,
        )

        return dataset_inputs


@dataclass
class InstantNGP(DataParser):
    """Instant NGP Dataset

    Args:
        data_directory: Location of data
        scale_factor: How much to scale the camera origins by.
        downscale_factor: How much to downscale images. Defaults to 1.
        scene_scale: How much to scale the scene. Defaults to 0.33
    """

    config: cfg.InstantNGPDataParserConfig

    def _generate_dataset_inputs(self, split="train"):

        abs_dir = get_absolute_path(self.config.data_directory)

        meta = load_from_json(abs_dir / "transforms.json")
        image_filenames = []
        poses = []
        num_skipped_image_filenames = 0
        for frame in meta["frames"]:
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
        poses = np.array(poses).astype(np.float32)
        poses[:3, 3] *= self.config.scene_scale

        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        k1 = float(meta["k1"])
        k2 = float(meta["k2"])
        p1 = float(meta["p1"])
        p2 = float(meta["p2"])
        image_height = int(meta["h"])
        image_width = int(meta["w"])

        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        distortion_params = camera_utils.get_distortion_params(k1=k1, k2=k2, p1=p1, p2=p2)

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = meta["aabb_scale"]
        scene_bounds = SceneBounds(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=image_height,
            width=image_width,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        # TODO(ethan): add alpha background color
        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            downscale_factor=self.config.downscale_factor,
            cameras=cameras,
            scene_bounds=scene_bounds,
        )

        return dataset_inputs


@dataclass
class Mipnerf360(DataParser):
    """MipNeRF 360 Dataset

    Args:
        data_directory: Location of data
        downscale_factor: How much to downscale images. Defaults to 1.
        val_skip: 1/val_skip images to use for validation. Defaults to 8.
        auto_scale: Scale based on pose bounds. Defaults to True.
        aabb_scale: Scene scale, Defaults to 1.0.
    """

    config: cfg.MipNerf360DataParserConfig

    @classmethod
    def normalize_orientation(cls, poses: np.ndarray):
        """Set the _up_ direction to be in the positive Y direction.

        Args:
            poses: Numpy array of poses.
        """
        poses_orig = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        center = poses[:, :3, 3].mean(0)
        vec2 = poses[:, :3, 2].sum(0) / np.linalg.norm(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        vec0 = np.cross(up, vec2) / np.linalg.norm(np.cross(up, vec2))
        vec1 = np.cross(vec2, vec0) / np.linalg.norm(np.cross(vec2, vec0))
        c2w = np.stack([vec0, vec1, vec2, center], -1)  # [3, 4]
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # [4, 4]
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # [BS, 1, 4]
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # [BS, 4, 4]
        poses = np.linalg.inv(c2w) @ poses
        poses_orig[:, :3, :4] = poses[:, :3, :4]
        return poses_orig

    def _generate_dataset_inputs(self, split="train"):
        abs_dir = get_absolute_path(self.config.data_directory)
        image_dir = "images"
        if self.config.downscale_factor > 1:
            image_dir += f"_{self.config.downscale_factor}"
        image_dir = abs_dir / image_dir
        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} doesn't exist")

        valid_formats = [".jpg", ".png"]
        image_filenames = []
        for f in image_dir.iterdir():
            ext = f.suffix
            if ext.lower() not in valid_formats:
                continue
            image_filenames.append(image_dir / f)
        image_filenames = sorted(image_filenames)
        num_images = len(image_filenames)

        poses_data = np.load(abs_dir / "poses_bounds.npy")
        poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
        bounds = poses_data[:, -2:].transpose([1, 0])

        if num_images != poses.shape[0]:
            raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

        idx_test = np.arange(num_images)[:: self.config.val_skip]
        idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
        idx = idx_train if split == "train" else idx_test

        image_filenames = np.array(image_filenames)[idx]
        poses = poses[idx]

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]

        poses[:, :2, 4] = np.array([image_height, image_width])
        poses[:, 2, 4] = poses[:, 2, 4] * 1.0 / self.config.downscale_factor

        # Reorder pose to match our convention
        poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

        # Center poses and rotate. (Compute up from average of all poses)
        poses = self.normalize_orientation(poses)

        # Scale factor used in mipnerf
        if self.config.auto_scale:
            scale_factor = 1 / (np.min(bounds) * 0.75)
            poses[:, :3, 3] *= scale_factor
            bounds *= scale_factor

        # Center poses
        poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, :], axis=0)[:, 3]

        focal_length = poses[0, -1, -1]

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3, :4])  # camera to world transform
        num_cameras = len(image_filenames)
        num_intrinsics_params = 3
        intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
        intrinsics *= torch.tensor([cx, cy, focal_length])

        aabb = torch.tensor([[-4, -4, -4], [4, 4, 4]], dtype=torch.float32) * self.config.aabb_scale
        scene_bounds = SceneBounds(aabb=aabb)

        cameras = Cameras(
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            downscale_factor=1,
            cameras=cameras,
            scene_bounds=scene_bounds,
        )

        return dataset_inputs


@dataclass
class Record3D(DataParser):
    """Record3D Dataset

    Args:
        data_directory: Location of data
        downscale_factor: How much to downscale images. Defaults to 1.
        val_skip: 1/val_skip images to use for validation. Defaults to 8.
        aabb_scale: Scene scale, Defaults to 4.0.
        max_dataset_size: Max number of images to train on. If the dataset has
            more, images will be sampled approximately evenly. Defaults to 150.
    """

    config: cfg.Record3DDataParserConfig

    def _generate_dataset_inputs(self, split: str = "train") -> DatasetInputs:
        abs_dir = get_absolute_path(self.config.data_directory)

        image_dir = abs_dir / "rgb"

        if not image_dir.exists():
            raise ValueError(f"Image directory {image_dir} doesn't exist")

        image_filenames = []
        for f in image_dir.iterdir():
            if f.stem.isdigit():  # removes possible duplicate images (for example, 123(3).jpg)
                image_filenames.append(image_dir / f)

        image_filenames = sorted(image_filenames, key=lambda fn: int(fn.stem))
        image_filenames = np.array(image_filenames)
        num_images = len(image_filenames)

        metadata_path = abs_dir / "metadata.json"
        metadata_dict = load_from_json(metadata_path)

        poses_data = np.array(metadata_dict["poses"])
        # (N, 3, 4)
        poses = np.concatenate(
            [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
            axis=-1,
        ).astype(np.float32)

        if num_images > self.config.max_dataset_size:
            # Evenly select max_dataset_size images from dataset, including first
            # and last indices.
            idx = np.round(np.linspace(0, num_images - 1, self.config.max_dataset_size)).astype(int)
            poses = poses[idx]
            image_filenames = image_filenames[idx]
            num_images = len(image_filenames)

        # Normalization similar to Mipnerf360
        poses = Mipnerf360.normalize_orientation(poses)

        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2).astype(np.float32)

        rotation_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        poses = rotation_matrix @ poses

        idx_test = np.arange(num_images)[:: self.config.val_skip]
        idx_train = np.array([i for i in np.arange(num_images) if i not in idx_test])
        idx = idx_train if split == "train" else idx_test
        if num_images != poses.shape[0]:
            raise RuntimeError(f"Different number of images ({num_images}), and poses ({poses.shape[0]})")

        image_filenames = image_filenames[idx]
        poses = poses[idx]

        # Centering poses
        poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, 3], axis=0)

        camera_to_world = torch.from_numpy(poses[:, :3, :4])  # camera to world transform

        # Camera intrinsics
        K = np.array(metadata_dict["K"]).reshape((3, 3)).T
        focal_length = K[0, 0]

        H = metadata_dict["h"]
        W = metadata_dict["w"]

        # TODO(akristoffersen): The metadata dict comes with principle points,
        # but caused errors in image coord indexing. Should update once that is fixed.
        cx, cy = W / 2, H / 2

        num_cameras = len(image_filenames)
        num_intrinsics_params = 3
        intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
        intrinsics *= torch.tensor([cx, cy, focal_length])

        # scene_bounds = SceneBounds.from_camera_poses(camera_to_world, self.config.aabb_scale)
        aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32) * self.config.aabb_scale
        scene_bounds = SceneBounds(aabb=aabb)

        cameras = Cameras(
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            downscale_factor=self.config.downscale_factor,
            cameras=cameras,
            scene_bounds=scene_bounds,
        )

        return dataset_inputs


@dataclass
class Friends(DataParser):
    """Friends Dataset

    Args:
        data_directory: Location of data
        downscale_factor: How much to downscale images. Defaults to 1.
        include_semantics: whether or not to include the semantics. Defaults to False.
        include_point_cloud: whether or not to include the point cloud. Defaults to False.
    """

    data_directory: Path
    downscale_factor: int = 1
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
            points_3d = read_pointsTD_binary(abs_dir / "colmap" / "points3D.bin")
            xyz = torch.tensor(np.array([p_value.xyz for p_id, p_value in points_3d.items()])).float()
            rgb = torch.tensor(np.array([p_value.rgb for p_id, p_value in points_3d.items()])).float()
            xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], -1)
            xyz = (xyz_h @ transposed_point_cloud_transform.T)[..., :3]
            xyz = (xyz - box_center) * box_scale_factor  # center and rescale
            point_cloud.xyz = xyz
            point_cloud.rgb = rgb

        cameras = Cameras(
            fx=focal_lengths / self.downscale_factor,
            fy=focal_lengths / self.downscale_factor,
            cx=cx / self.downscale_factor,
            cy=cy / self.downscale_factor,
            camera_to_worlds=camera_to_world,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataset_inputs = DatasetInputs(
            image_filenames=image_filenames,
            downscale_factor=self.downscale_factor,
            cameras=cameras,
            semantics=semantics,
            point_cloud=point_cloud,
            scene_bounds=scene_bounds,
        )
        return dataset_inputs
