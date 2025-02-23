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
"""Helper utils for processing Stray Scanner data into the nerfstudio format."""

import json
import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

from nerfstudio.process_data import process_data_utils
from nerfstudio.process_data.process_data_utils import (
    CAMERA_MODELS,
    convert_video_to_images,
)
from nerfstudio.utils import io

import pandas as pd
import cv2
import os
import numpy as np

CONSOLE = Console(width=120)


def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix

def strayscan_to_json(
    image_filenames: List[Path],
    intrinsics_file_base_path: Path,
    num_frames: int,
    num_frames_target: int,
    output_dir: Path,
) -> List[str]:
    """Convert strayscanner data into a nerfstudio dataset.

    Args:
        image_filenames: List of paths to the original images.
        depth_filenames: List of paths to the original depth maps.
        cameras_dir: Path to the polycam cameras directory.
        output_dir: Path to the output directory.
        min_blur_score: Minimum blur score to use an image. Images below this value will be skipped.
        crop_border_pixels: Number of pixels to crop from each border of the image.

    Returns:
        Summary of the conversion.
    """

    data = {}
    # Extracting base path from image paths
    # base_dir_path = image_filenames[0].split("images/")[0]
    base_dir_path = intrinsics_file_base_path
    # Extracting camera matrix data and Odometry data
    path_to_camera_matrix = base_dir_path + "camera_matrix.csv"
    cam_matrix = pd.read_csv(path_to_camera_matrix, header=None)
    path_to_odometry_data = base_dir_path + "odometry.csv"
    odometry_data = pd.read_csv(path_to_odometry_data)

    # # Getting c_x, c_y, f_x, f_y from camera_matrix.csv
    # data["camera_model"]= CAMERA_MODELS["perspective"].value TO ADD
    data["fl_x"] = cam_matrix.loc[0][0]
    data["fl_y"] = cam_matrix.loc[1][1]
    data["cx"] = cam_matrix.loc[0][2]
    data["cy"] = cam_matrix.loc[1][2]
    # reading a image for H, W
    temp_img = cv2.imread(str(image_filenames[0]))
    H, W = temp_img.shape[:2]
    data["h"] = H
    data["w"] = W

    # populating the data related to each frame R, t, image_path
    spacing = num_frames // num_frames_target

    print("SPACING ", spacing)

    if spacing < 1:
        spacing = 1

    frames = []
    for i, image_path in enumerate(image_filenames):

        frame = {}
        frame["file_path"] = "images/" + os.path.basename(image_path.as_posix())
        rotation_matrix = quaternion_to_rotation_matrix(odometry_data.loc[spacing * i][5:]).tolist()
        translation = odometry_data.loc[spacing * i][2:5]
        translation = np.array(translation).reshape(3, 1)
        w2c = np.concatenate([rotation_matrix, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = w2c
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        frame["transform_matrix"] = c2w.tolist()
        frames.append(frame)

    data["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summary = []

    return summary