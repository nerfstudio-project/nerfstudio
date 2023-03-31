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


def _find_param(calib_xml: ET.Element, param_name: str):
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
    if not calibrated_sensors:
        raise ValueError("No calibrated sensor found in Metashape XML")
    sensor_type = [s.get("type") for s in calibrated_sensors]
    if sensor_type.count(sensor_type[0]) != len(sensor_type):
        raise ValueError("All Metashape sensors do not have the same sensor type")
    data = {}
    if sensor_type[0] == "frame":
        data["camera_model"] = CAMERA_MODELS["perspective"].value
    elif sensor_type[0] == "fisheye":
        data["camera_model"] = CAMERA_MODELS["fisheye"].value
    elif sensor_type[0] == "spherical":
        data["camera_model"] = CAMERA_MODELS["equirectangular"].value
    else:
        # Cylindrical and RPC sensor types are not supported
        raise ValueError(f"Unsupported Metashape sensor type '{sensor_type[0]}'")

    sensor_dict = {}
    for sensor in calibrated_sensors:
        s = {}
        resolution = sensor.find("resolution")
        assert resolution is not None, "Resolution not found in Metashape xml"
        s["w"] = int(resolution.get("width"))  # type: ignore
        s["h"] = int(resolution.get("height"))  # type: ignore

        calib = sensor.find("calibration")
        f = calib.find("f")
        if f is not None:
            s["fl_x"] = s["fl_y"] = float(f.text)  # type: ignore
        s["cx"] = _find_param(calib, "cx") + s["w"] / 2.0  # type: ignore
        s["cy"] = _find_param(calib, "cy") + s["h"] / 2.0  # type: ignore

        s["k1"] = _find_param(calib, "k1")
        s["k2"] = _find_param(calib, "k2")
        s["k3"] = _find_param(calib, "k3")
        s["k4"] = _find_param(calib, "k4")
        s["p1"] = _find_param(calib, "p1")
        s["p2"] = _find_param(calib, "p2")

        sensor_dict[sensor.get("id")] = s

    components = chunk.find("components")
    component_dict = {}
    for component in components:
        transform = component.find("transform")
        if transform is not None:
            rotation = transform.find("rotation")
            if rotation is None:
                r = np.eye(3)
            else:
                r = np.array([float(x) for x in rotation.text.split()]).reshape((3, 3))
            translation = transform.find("translation")
            if translation is None:
                t = np.zeros(3)
            else:
                t = np.array([float(x) for x in translation.text.split()])
            scale = transform.find("scale")
            if scale is None:
                s = 1.0
            else:
                s = float(scale.text)

            m = np.eye(4)
            m[:3, :3] = r
            m[:3, 3] = t / s
            component_dict[component.get("id")] = m

    frames = []
    cameras = chunk.find("cameras")
    assert cameras is not None, "Cameras not found in Metashape xml"
    num_skipped = 0
    for camera in cameras:
        frame = {}
        camera_label = camera.get("label")
        if camera_label not in image_filename_map:
            # Labels sometimes have a file extension. Try without the extension.
            # (maybe it's just a '.' in the image name)
            camera_label = camera_label.split(".")[0]  # type: ignore
            if camera_label not in image_filename_map:
                continue
        frame["file_path"] = image_filename_map[camera_label].as_posix()

        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            # this should only happen when we have a sensor that doesn't have calibration
            if verbose:
                CONSOLE.print(f"Missing sensor calibration for {camera.get('label')}, Skipping")
            num_skipped += 1
            continue
        # Add all sensor parameters to this frame.
        frame.update(sensor_dict[sensor_id])

        if camera.find("transform") is None:
            if verbose:
                CONSOLE.print(f"Missing transforms data for {camera.get('label')}, Skipping")
            num_skipped += 1
            continue
        transform = np.array([float(x) for x in camera.find("transform").text.split()]).reshape((4, 4))
        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform
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
