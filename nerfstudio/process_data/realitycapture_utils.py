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

"""Helper utils for processing reality capture data into the nerfstudio format."""

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils.rich_utils import CONSOLE


def realitycapture_to_json(
    image_filename_map: Dict[str, Path],
    csv_filename: Path,
    ply_filename: Optional[Path],
    output_dir: Path,
    verbose: bool = False,
) -> List[str]:
    """Convert RealityCapture data into a nerfstudio dataset.
    See https://dev.epicgames.com/community/learning/knowledge-base/vzwB/capturing-reality-realitycapture-xmp-camera-math

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

    missing_image_data = 0

    for i, name in enumerate(cameras["#name"]):
        basename = name.rpartition(".")[0]
        if basename not in image_filename_map:
            if verbose:
                CONSOLE.print(f"Missing image for camera data {basename}, Skipping")
            missing_image_data += 1
            continue

        frame = {}
        img = np.array(Image.open(output_dir / image_filename_map[basename]))
        height, width, _ = img.shape
        frame["h"] = int(height)
        frame["w"] = int(width)
        frame["file_path"] = image_filename_map[basename].as_posix()
        # reality capture uses the 35mm equivalent focal length
        # See https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
        scale = max(width, height)
        frame["fl_x"] = float(cameras["f"][i]) * scale / 36.0
        frame["fl_y"] = float(cameras["f"][i]) * scale / 36.0
        frame["cx"] = float(cameras["px"][i]) * scale + width / 2.0
        frame["cy"] = float(cameras["py"][i]) * scale + height / 2.0
        frame["k1"] = float(cameras["k1"][i])
        frame["k2"] = float(cameras["k2"][i])
        frame["k3"] = float(cameras["k3"][i])
        frame["k4"] = float(cameras["k4"][i])
        frame["p1"] = float(cameras["t1"][i])
        frame["p2"] = float(cameras["t2"][i])

        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        rot = _get_rotation_matrix(-float(cameras["heading"][i]), float(cameras["pitch"][i]), float(cameras["roll"][i]))

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = np.array([float(cameras["x"][i]), float(cameras["y"][i]), float(cameras["alt"][i])])

        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)
    data["frames"] = frames

    if ply_filename is not None:
        shutil.copy(ply_filename, output_dir / "sparse_pc.ply")
        data["ply_file_path"] = "sparse_pc.ply"

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
