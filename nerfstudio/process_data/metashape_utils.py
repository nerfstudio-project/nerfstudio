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

"""Helper utils for processing metashape data into the nerfstudio format."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console

from nerfstudio.process_data.process_data_utils import CAMERA_MODELS

CONSOLE = Console(width=120)


def _find_distortion_param(calib_xml: ET.Element, param_name: str):
    param = calib_xml.find(param_name)
    if param is not None:
        return float(param.text)  # type: ignore
    return 0.0


def metashape_to_json(  # pylint: disable=too-many-statements
    image_filename_map: Dict[str, Path],
    xml_filename: Path,
    output_dir: Path,
    verbose: bool = False,
) -> List[str]:
    """Convert Metashape data into a nerfstudio dataset.

    Args:
        image_filename_map: Mapping of original image filenames to their saved locations.
        xml_filename: Path to the metashape cameras xml file.
        output_dir: Path to the output directory.
        verbose: Whether to print verbose output.

    Returns:
        Summary of the conversion.
    """

    xml_tree = ET.parse(xml_filename)
    root = xml_tree.getroot()
    chunk = root[0]
    sensors = chunk.find("sensors")

    # TODO Add support for per-frame intrinsics
    if sensors is None:
        raise ValueError("No sensors found")

    calibrated_sensors = [sensor for sensor in sensors if sensor.find("calibration")]
    if len(calibrated_sensors) != 1:
        raise ValueError("Only one sensor is supported for now")

    sensor = calibrated_sensors[0]

    data = {}

    assert sensor is not None, "Sensor not found in Metashape XML"
    resolution = sensor.find("resolution")
    assert resolution is not None, "Resolution not found in Metashape xml"
    data["w"] = int(resolution.get("width"))  # type: ignore
    data["h"] = int(resolution.get("height"))  # type: ignore

    calib = sensor.find("calibration")
    assert calib is not None, "Calibration not found in Metashape xml"
    data["fl_x"] = float(calib.find("f").text)  # type: ignore
    data["fl_y"] = float(calib.find("f").text)  # type: ignore
    data["cx"] = float(calib.find("cx").text) + data["w"] / 2.0  # type: ignore
    data["cy"] = float(calib.find("cy").text) + data["h"] / 2.0  # type: ignore

    data["k1"] = _find_distortion_param(calib, "k1")
    data["k2"] = _find_distortion_param(calib, "k2")
    data["k3"] = _find_distortion_param(calib, "k3")
    data["k4"] = _find_distortion_param(calib, "k4")
    data["p1"] = _find_distortion_param(calib, "p1")
    data["p2"] = _find_distortion_param(calib, "p2")

    data["camera_model"] = CAMERA_MODELS["perspective"].value

    frames = []
    cameras = chunk.find("cameras")
    assert cameras is not None, "Cameras not found in Metashape xml"
    num_skipped = 0
    for camera in cameras:
        frame = {}
        # Labels sometimes have a file extension. We remove it for consistency.
        camera_label = camera.get("label").split(".")[0]  # type: ignore
        if camera_label not in image_filename_map:
            continue
        frame["file_path"] = image_filename_map[camera_label].as_posix()

        if camera.get("sensor_id") != sensor.get("id"):
            # this should only happen when we have a sensor that doesn't have calibration
            if verbose:
                CONSOLE.print(f"Missing sensor calibration for {camera.get('label')}, Skipping")
            num_skipped += 1
            continue

        if camera.find("transform") is None:
            if verbose:
                CONSOLE.print(f"Missing transforms data for {camera.get('label')}, Skipping")
            num_skipped += 1
            continue
        t = [float(x) for x in camera.find("transform").text.split()]  # type: ignore
        transform = np.array(
            [
                [t[8], -t[9], -t[10], t[11]],
                [t[0], -t[1], -t[2], t[3]],
                [t[4], -t[5], -t[6], t[7]],
                [t[12], -t[13], -t[14], t[15]],
            ]
        )
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
