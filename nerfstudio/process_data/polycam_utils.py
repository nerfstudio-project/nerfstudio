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

import json
import sys
from pathlib import Path
from typing import List

from rich.console import Console

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import io

CONSOLE = Console(width=120)


def polycam_to_json(
    image_filenames: List[Path],
    cameras_dir: Path,
    output_dir: Path,
    min_blur_score: float = 0.0,
    crop_border_pixels: int = 0,
) -> List[str]:
    """Convert Polycam data into a nerfstudio dataset.

    Args:
        image_filenames: List of paths to the original images.
        cameras_dir: Path to the polycam cameras directory.
        output_dir: Path to the output directory.
        min_blur_score: Minimum blur score to use an image. Images below this value will be skipped.
        crop_border_pixels: Number of pixels to crop from each border of the image.

    Returns:
        Summary of the conversion.
    """
    data = {}
    data["camera_model"] = CAMERA_MODELS["perspective"].value
    # Needs to be a string for camera_utils.auto_orient_and_center_poses
    data["orientation_override"] = "none"

    frames = []
    skipped_frames = 0
    for i, image_filename in enumerate(image_filenames):
        json_filename = cameras_dir / f"{image_filename.stem}.json"
        frame_json = io.load_from_json(json_filename)
        if "blur_score" in frame_json and frame_json["blur_score"] < min_blur_score:
            skipped_frames += 1
            continue
        frame = {}
        frame["fl_x"] = frame_json["fx"]
        frame["fl_y"] = frame_json["fy"]
        frame["cx"] = frame_json["cx"] - crop_border_pixels
        frame["cy"] = frame_json["cy"] - crop_border_pixels
        frame["w"] = frame_json["width"] - crop_border_pixels * 2
        frame["h"] = frame_json["height"] - crop_border_pixels * 2
        frame["file_path"] = f"./images/frame_{i+1:05d}{image_filename.suffix}"
        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        frame["transform_matrix"] = [
            [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
            [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
            [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append(frame)
    data["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summary = []
    if skipped_frames > 0:
        summary.append(f"Skipped {skipped_frames} frames due to low blur score.")
    summary.append(f"Final dataset is {len(image_filenames) - skipped_frames} frames.")

    if len(image_filenames) - skipped_frames == 0:
        CONSOLE.print("[bold red]No images remain after filtering, exiting")
        sys.exit(1)

    return summary
