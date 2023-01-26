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

"""Helper utils for processing polycam data into the nerfstudio format."""

import csv
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from rich.console import Console

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS

CONSOLE = Console(width=120)


def realitycapture_to_json(
    image_filename_map: List[Path],
    csv_filename: Path,
    output_dir: Path,
    verbose: bool = False,
) -> List[str]:
    """Convert RealityCapture data into a nerfstudio dataset.

    Args:
        image_filenames: List of paths to the original images.
        csv_filename: Path to the csv file containing the camera poses.
        output_dir: Path to the output directory.
        verbose: Whether to print verbose output.

    Returns:
        Summary of the conversion.
    """
    data = {}
    data["camera_model"] = CAMERA_MODELS["perspective"].value
    # Needs to be a string for camera_utils.auto_orient_and_center_poses
    data["orientation_override"] = "none"

    frames = []

    with open(csv_filename, encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        cameras = {}
        for row in reader:
            for column, value in row.items():
                cameras.setdefault(column, []).append(value)

    for name in cameras["#name"]:
        camera_label = name.split(".")[0]
        if camera_label in image_filename_map:
            img = np.array(Image.open(output_dir / image_filename_map[camera_label]))
            break

    height, width, _ = img.shape

    data["h"] = int(height)
    data["w"] = int(width)

    missing_image_data = 0

    for i, name in enumerate(cameras["#name"]):
        basename = name.split(".")[0]
        if basename not in image_filename_map:
            if verbose:
                CONSOLE.print(f"Missing image for camera data {basename}, Skipping")
            missing_image_data += 1
            continue
        frame = {}
        frame["file_path"] = image_filename_map[basename].as_posix()
        frame["fl_x"] = float(cameras["f"][i]) * max(width, height) / 36
        frame["fl_y"] = float(cameras["f"][i]) * max(width, height) / 36
        # TODO: Unclear how to get the principal point from RealityCapture, here a guess...
        frame["cx"] = float(cameras["px"][i]) / 36.0 + width / 2.0
        frame["cy"] = float(cameras["py"][i]) / 36.0 + height / 2.0
        # TODO: Not sure if RealityCapture uses this distortion model
        frame["k1"] = cameras["k1"][i]
        frame["k2"] = cameras["k2"][i]
        frame["k3"] = cameras["k3"][i]
        frame["k4"] = cameras["k4"][i]
        frame["p1"] = cameras["t1"][i]
        frame["p2"] = cameras["t2"][i]

        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        rot = _get_rotation_matrix(-float(cameras["heading"][i]), float(cameras["pitch"][i]), float(cameras["roll"][i]))

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = np.array([float(cameras["x"][i]), float(cameras["y"][i]), float(cameras["alt"][i])])

        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)
    data["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summary = []
    if missing_image_data > 0:
        summary.append(f"Missing image data for {missing_image_data} cameras.")
    if len(frames) < len(image_filename_map):
        summary.append(f"Missing camera data for {len(image_filename_map) - len(frames)} frames.")
    summary.append(f"Final dataset is {len(frames)} frames.")

    return summary


def _get_rotation_matrix(yaw, pitch, roll):
    """Returns a rotation matrix given euler angles."""

    s_yaw = np.sin(np.deg2rad(yaw))
    c_yaw = np.cos(np.deg2rad(yaw))
    s_pitch = np.sin(np.deg2rad(pitch))
    c_pitch = np.cos(np.deg2rad(pitch))
    s_roll = np.sin(np.deg2rad(roll))
    c_roll = np.cos(np.deg2rad(roll))

    rot_x = np.array([[1, 0, 0], [0, c_pitch, -s_pitch], [0, s_pitch, c_pitch]])
    rot_y = np.array([[c_roll, 0, s_roll], [0, 1, 0], [-s_roll, 0, c_roll]])
    rot_z = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])

    return rot_z @ rot_x @ rot_y
