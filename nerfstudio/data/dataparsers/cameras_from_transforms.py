from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from rich.console import Console

from nerfstudio.nerfstudio.cameras import camera_utils
from nerfstudio.nerfstudio.cameras.cameras import (
    CAMERA_MODEL_TO_TYPE,
    Cameras,
    CameraType,
)
from nerfstudio.nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
from nerfstudio.nerfstudio.data.scene_box import SceneBox
from nerfstudio.nerfstudio.defaults import SPLIT_MODE_ALL
from nerfstudio.nerfstudio.utils.io import load_from_json

downscale_factor: Optional[int] = None

CONSOLE = Console(width=120, no_color=True)


def generate_dataparser_outputs_static(
    data_or_transforms: Path,
    indices_file: Path | None,
    center_poses,
    auto_scale_poses,
    scene_scale,
    orientation_method,
    split="train",
):
    """This exists so we can easily extract cameras from transforms.json and (dataset's images)"""
    # pylint: disable=too-many-statements

    if data_or_transforms.suffix == ".json":
        meta = load_from_json(data_or_transforms)
        data_dir = data_or_transforms.parent
    else:
        meta = load_from_json(data_or_transforms / "transforms.json")
        data_dir = data_or_transforms

    indices_json: None | dict[str, str] = None
    if indices_file is not None:
        indices_file_path = indices_file
        if indices_file_path.is_file():
            indices_json = load_from_json(indices_file_path)
        else:
            raise FileNotFoundError(f"File {str(indices_file_path)} is not found.")

    image_filenames = []
    poses = []
    num_skipped_image_filenames = 0

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
        if distort_key in meta:
            distort_fixed = True
            break
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    for frame in meta["frames"]:
        filepath = PurePath(frame["file_path"])
        # TODO: matej
        filepath = Path(*filepath.parts[-2:])
        fname = Path(data_dir, filepath)
        if not fname.exists():
            num_skipped_image_filenames += 1
            CONSOLE.log(f"Skiping image {str(fname)} because it doesn't exist.")
            continue

        indices_json_key = str(Path("images", fname.name))
        if indices_json is not None and (
            indices_json_key not in indices_json
            or indices_json[indices_json_key] == "ignore"
        ):
            num_skipped_image_filenames += 1
            # CONSOLE.log(
            #     f"Skiping image {str(fname)} because it doesn't exist in {str(indices_file_path)}"
            # )
            continue

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
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

        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))

    if num_skipped_image_filenames >= 0:
        CONSOLE.log(
            f"Skipping {num_skipped_image_filenames} files in dataset split {split}."
        )
    assert (
        len(image_filenames) != 0
    ), """
    No image files found. 
    You should check the file_paths in the transforms.json file to make sure they are correct.
    """

    if split in ["val", "test"]:
        split_strategy = ["val", "test"]
    elif split in ["train"]:
        split_strategy = ["train"]
    elif split in [SPLIT_MODE_ALL]:
        split_strategy = ["train", "val", "test"]
    else:
        raise ValueError(f"Split can't be '{split}'")

    # assert len(indices_json) == len(
    #     image_filenames
    # ), "Indices file and number of image files in the directory should be equal."

    if indices_json:
        _indices = []

        for image_path, image_split in indices_json.items():

            if image_split == "ignore":
                continue
            if image_split not in split_strategy:
                continue

            i = image_filenames.index(Path(data_dir, image_path))
            _indices.append(i)

        indices = np.array(sorted(_indices), dtype=np.int64)
    else:
        indices = np.arange(0, len(image_filenames))

    assert len(indices), "Indices array is empty"

    if "orientation_override" in meta:
        orientation_method = meta["orientation_override"]
        CONSOLE.log(
            f"[yellow] Dataset is overriding orientation method to {orientation_method}"
        )

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        poses,
        method=orientation_method,
        center_poses=center_poses,
    )

    # Scale poses
    scale_factor = 1.0
    if auto_scale_poses:
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    scale_factor *= scale_factor

    poses[:, :3, 3] *= scale_factor

    # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
    image_filenames = [image_filenames[i] for i in indices]
    poses = poses[indices]

    # in x,y,z order
    # assumes that the scene is centered at the origin
    aabb_scale = scene_scale
    scene_box = SceneBox(
        aabb=torch.tensor(
            [
                [-aabb_scale, -aabb_scale, -aabb_scale],
                [aabb_scale, aabb_scale, aabb_scale],
            ],
            dtype=torch.float32,
        )
    )

    if "camera_model" in meta:
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
    else:
        camera_type = CameraType.PERSPECTIVE

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    fx = (
        float(meta["fl_x"])
        if fx_fixed
        else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
    )
    fy = (
        float(meta["fl_y"])
        if fy_fixed
        else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
    )
    cx = (
        float(meta["cx"])
        if cx_fixed
        else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
    )
    cy = (
        float(meta["cy"])
        if cy_fixed
        else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
    )
    height = (
        int(meta["h"])
        if height_fixed
        else torch.tensor(height, dtype=torch.int32)[idx_tensor]
    )
    width = (
        int(meta["w"])
        if width_fixed
        else torch.tensor(width, dtype=torch.int32)[idx_tensor]
    )
    if distort_fixed:
        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )
    else:
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

    downscale_factor = 1
    cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

    dataparser_outputs = DataparserOutputs(
        image_filenames=image_filenames,
        cameras=cameras,
        scene_box=scene_box,
        mask_filenames=None,
        dataparser_scale=scale_factor,
        dataparser_transform=transform_matrix,
        metadata={},
    )
    return dataparser_outputs
