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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.prompt import Confirm

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command

MAX_AUTO_RESOLUTION = 1600


@dataclass
class ColmapDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: ColmapDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    downscale_rounding_mode: Literal["floor", "round", "ceil"] = "floor"
    """How to round downscale image height and Image width."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    assume_colmap_world_coordinate_convention: bool = True
    """Colmap optimized world often have y direction of the first camera pointing towards down direction,
    while nerfstudio world set z direction to be up direction for viewer. Therefore, we usually need to apply an extra
    transform when orientation_method=none. This parameter has no effects if orientation_method is set other than none.
    When this parameter is set to False, no extra transform is applied when reading data from colmap.
    """
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    masks_path: Optional[Path] = None
    """Path to masks directory. If not set, masks are not loaded."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""


class ColmapDataParser(DataParser):
    """COLMAP DatasetParser.
    Expects a folder with the following structure:
        images/ # folder containing images used to create the COLMAP model
        sparse/0 # folder containing the COLMAP reconstruction (either TEXT or BINARY format)
        masks/ # (OPTIONAL) folder containing masks for each image
        depths/ # (OPTIONAL) folder containing depth maps for each image
    The paths can be different and can be specified in the config. (e.g., sparse/0 -> sparse)
    Currently, most COLMAP camera models are supported except for the FULL_OPENCV and THIN_PRISM_FISHEYE models.

    The dataparser loads the downscaled images from folders with `_{downscale_factor}` suffix.
    If these folders do not exist, the user can choose to automatically downscale the images and
    create these folders.

    The loader is compatible with the datasets processed using the ns-process-data script and
    can be used as a drop-in replacement. It further supports datasets like Mip-NeRF 360 (although
    in the case of Mip-NeRF 360 the downsampled images may have a different resolution because they
    use different rounding when computing the image resolution).
    """

    config: ColmapDataParserConfig

    def __init__(self, config: ColmapDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        cameras = {}
        frames = []
        camera_model = None

        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())
        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            if self.config.assume_colmap_world_coordinate_convention:
                # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
                c2w = c2w[np.array([0, 2, 1, 3]), :]
                c2w[2, :] *= -1

            frame = {
                "file_path": (self.config.data / self.config.images_path / im_data.name).as_posix(),
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            if self.config.masks_path is not None:
                frame["mask_path"] = (
                    (self.config.data / self.config.masks_path / im_data.name).with_suffix(".png").as_posix()
                )
            if self.config.depths_path is not None:
                frame["depth_path"] = (
                    (self.config.data / self.config.depths_path / im_data.name).with_suffix(".png").as_posix()
                )
            frames.append(frame)
            if camera_model is not None:
                assert camera_model == frame["camera_model"], "Multiple camera models are not supported"
            else:
                camera_model = frame["camera_model"]

        out = {}
        out["frames"] = frames
        if self.config.assume_colmap_world_coordinate_convention:
            # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()
        out["camera_model"] = camera_model
        assert len(frames) > 0, "No images found in the colmap model"
        return out

    def _get_image_indices(self, image_filenames, split):
        has_split_files_spec = (
            (self.config.data / "train_list.txt").exists()
            or (self.config.data / "test_list.txt").exists()
            or (self.config.data / "validation_list.txt").exists()
        )
        if (self.config.data / f"{split}_list.txt").exists():
            CONSOLE.log(f"Using {split}_list.txt to get indices for split {split}.")
            with (self.config.data / f"{split}_list.txt").open("r", encoding="utf8") as f:
                filenames = f.read().splitlines()
            # Validate split first
            split_filenames = set(self.config.data / self.config.images_path / x for x in filenames)
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {set(map(str, unmatched_filenames))}."
                )

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

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

        cameras.rescale_output_resolution(
            scaling_factor=1.0 / downscale_factor, scale_rounding_mode=self.config.downscale_rounding_mode
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_path}")
        points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))
        points3D_num_points = torch.tensor([len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64)
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(np.array([p.error for p in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": points3D_num_points,
        }
        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
            else:
                raise ValueError(f"Could not find images.txt or images.bin in {colmap_path}")
            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(max_num_points, self.config.max_2D_matches_per_3D_point)
            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    # Randomly sample 2D matches
                    idxs = np.argsort(p.error)[: self.config.max_2D_matches_per_3D_point]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [im_id_to_image[im_id].xys[pt_idx] for im_id, pt_idx in zip(nids, nxy_ids)]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat((nids, torch.full((max_num_points - len(nids),), -1, dtype=torch.int64)))
                )
                points3D_image_xy.append(
                    torch.cat((nxy, torch.full((max_num_points - len(nxy), nxy.shape[-1]), 0, dtype=torch.float32)))
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def _downscale_images(
        self,
        paths,
        get_fname,
        downscale_factor: int,
        downscale_rounding_mode: str = "floor",
        nearest_neighbor: bool = False,
    ):
        def calculate_scaled_size(original_width, original_height, downscale_factor, mode="floor"):
            if mode == "floor":
                return math.floor(original_width / downscale_factor), math.floor(original_height / downscale_factor)
            elif mode == "round":
                return round(original_width / downscale_factor), round(original_height / downscale_factor)
            elif mode == "ceil":
                return math.ceil(original_width / downscale_factor), math.ceil(original_height / downscale_factor)
            else:
                raise ValueError("Invalid mode. Choose from 'floor', 'round', or 'ceil'.")

        with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            filepath = next(iter(paths))
            img = Image.open(filepath)
            w, h = img.size
            w_scaled, h_scaled = calculate_scaled_size(w, h, downscale_factor, downscale_rounding_mode)
            # Using %05d ffmpeg commands appears to be unreliable (skips images).
            for path in paths:
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                path_out = get_fname(path)
                path_out.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg_cmd = [
                    f'ffmpeg -y -noautorotate -i "{path}" ',
                    f"-q:v 2 -vf scale={w_scaled}:{h_scaled}{nn_flag} ",
                    f'"{path_out}"',
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd)

        CONSOLE.log("[bold green]:tada: Done downscaling images.")

    def _setup_downscale_factor(
        self, image_filenames: List[Path], mask_filenames: List[Path], depth_filenames: List[Path]
    ):
        """
        Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
        """

        def get_fname(parent: Path, filepath: Path) -> Path:
            """Returns transformed file name when downscale factor is applied"""
            rel_part = filepath.relative_to(parent)
            base_part = parent.parent / (str(parent.name) + f"_{self._downscale_factor}")
            return base_part / rel_part

        filepath = next(iter(image_filenames))
        if self._downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(filepath)
                w, h = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    df += 1

                self._downscale_factor = 2**df
                CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
            else:
                self._downscale_factor = self.config.downscale_factor
            if self._downscale_factor > 1 and not all(
                get_fname(self.config.data / self.config.images_path, fp).parent.exists() for fp in image_filenames
            ):
                # Downscaled images not found
                # Ask if user wants to downscale the images automatically here
                CONSOLE.print(
                    f"[bold red]Downscaled images do not exist for factor of {self._downscale_factor}.[/bold red]"
                )
                if Confirm.ask(
                    f"\nWould you like to downscale the images using '{self.config.downscale_rounding_mode}' rounding mode now?",
                    default=False,
                    console=CONSOLE,
                ):
                    # Install the method
                    self._downscale_images(
                        image_filenames,
                        partial(get_fname, self.config.data / self.config.images_path),
                        self._downscale_factor,
                        self.config.downscale_rounding_mode,
                        nearest_neighbor=False,
                    )
                    if len(mask_filenames) > 0:
                        assert self.config.masks_path is not None
                        self._downscale_images(
                            mask_filenames,
                            partial(get_fname, self.config.data / self.config.masks_path),
                            self._downscale_factor,
                            self.config.downscale_rounding_mode,
                            nearest_neighbor=True,
                        )
                    if len(depth_filenames) > 0:
                        assert self.config.depths_path is not None
                        self._downscale_images(
                            depth_filenames,
                            partial(get_fname, self.config.data / self.config.depths_path),
                            self._downscale_factor,
                            self.config.downscale_rounding_mode,
                            nearest_neighbor=True,
                        )
                else:
                    sys.exit(1)

        # Return transformed filenames
        if self._downscale_factor > 1:
            image_filenames = [get_fname(self.config.data / self.config.images_path, fp) for fp in image_filenames]
            if len(mask_filenames) > 0:
                assert self.config.masks_path is not None
                mask_filenames = [get_fname(self.config.data / self.config.masks_path, fp) for fp in mask_filenames]
            if len(depth_filenames) > 0:
                assert self.config.depths_path is not None
                depth_filenames = [get_fname(self.config.data / self.config.depths_path, fp) for fp in depth_filenames]
        assert isinstance(self._downscale_factor, int)
        return image_filenames, mask_filenames, depth_filenames, self._downscale_factor
