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

"""Helper utils for processing ODM data into the nerfstudio format."""

import json
from pathlib import Path
from typing import Dict, List
import os
import sys
import math

import numpy as np

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS


def rodrigues_vec_to_rotation_mat(rodrigues_vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        ident = np.eye(3, dtype=float)
        r_rT = np.array(
            [
                [r[0] * r[0], r[0] * r[1], r[0] * r[2]],
                [r[1] * r[0], r[1] * r[1], r[1] * r[2]],
                [r[2] * r[0], r[2] * r[1], r[2] * r[2]],
            ]
        )
        r_cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=float)
        rotation_mat = math.cos(theta) * ident + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat


def cameras2nerfds(
    image_filename_map: Dict[str, Path],
    cameras_file: Path,
    shots_file: Path,
    output_dir: Path,
    verbose: bool = False,
) -> List[str]:
    """Convert ODM cameras into a nerfstudio dataset.

    Args:
        image_filename_map: Mapping of original image filenames to their saved locations.
        shots_file: Path to ODM's shots.geojson
        output_dir: Path to the output directory.
        verbose: Whether to print verbose output.

    Returns:
        Summary of the conversion.
    """

    with open(cameras_file, "r", encoding="utf-8") as f:
        cameras = json.loads(f.read())
    with open(shots_file, "r", encoding="utf-8") as f:
        shots = json.loads(f.read())

    camera_ids = list(cameras.keys())
    if len(camera_ids) > 1:
        raise ValueError("Only one camera is supported")
    camera_id = camera_ids[0]
    camera = cameras[camera_id]
    data = {}
    if camera["projection_type"] in ["brown", "perspective"]:
        data["camera_model"] = CAMERA_MODELS["perspective"].value
    elif camera["projection_type"] in ["fisheye", "fisheye_opencv"]:
        data["camera_model"] = CAMERA_MODELS["fisheye"].value
    elif camera["projection_type"] in ["spherical", "equirectangular"]:
        data["camera_model"] = CAMERA_MODELS["equirectangular"].value
    else:
        raise ValueError("Unsupported ODM camera model: " + data["camera_model"])

    sensor_dict = {}
    s = {"w": int(camera["width"]), "h": int(camera["height"])}

    s["fl_x"] = camera.get("focal_x", camera.get("focal")) * max(s["w"], s["h"])
    s["fl_y"] = camera.get("focal_y", camera.get("focal")) * max(s["w"], s["h"])

    s["cx"] = camera["c_x"] + (s["w"] - 1.0) / 2.0
    s["cy"] = camera["c_y"] + (s["h"] - 1.0) / 2.0

    for p in ["k1", "k2", "p1", "p2", "k3"]:
        if p in camera:
            s[p] = camera[p]

        sensor_dict[camera_id] = s

    shots = shots["features"]
    shots_dict = {}
    for shot in shots:
        props = shot["properties"]
        filename = props["filename"]
        rotation = rodrigues_vec_to_rotation_mat(np.array(props["rotation"]) * -1)
        translation = np.array(props["translation"])

        m = np.eye(4)
        m[:3, :3] = rotation
        m[:3, 3] = translation

        name, ext = os.path.splitext(filename)
        shots_dict[name] = m

    frames = []
    num_skipped = 0

    for fname in shots_dict:
        transform = shots_dict[fname]
        if fname not in image_filename_map:
            num_skipped += 1
            continue

        frame = {}
        frame["file_path"] = image_filename_map[fname].as_posix()
        frame.update(sensor_dict[camera_id])

        transform = transform[[2, 0, 1, 3], :]
        transform[:, 1:3] *= -1
        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)

    data["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summary = []
    if num_skipped == 1:
        summary.append(f"{num_skipped} image skipped because it was missing its camera pose.")
    if num_skipped > 1:
        summary.append(f"{num_skipped} images were skipped because they were missing camera poses.")

    summary.append(f"Final dataset is {len(data['frames'])} frames.")

    return summary
