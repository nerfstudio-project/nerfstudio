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

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import open3d as o3d
import tyro
from PIL import Image

try:
    from projectaria_tools.core import mps
    from projectaria_tools.core.data_provider import VrsDataProvider, create_vrs_data_provider
    from projectaria_tools.core.mps.utils import filter_points_from_confidence
    from projectaria_tools.core.sophus import SE3
except ImportError:
    print("projectaria_tools import failed, please install with pip3 install projectaria-tools'[all]'")
    sys.exit(1)

ARIA_CAMERA_MODEL = "FISHEYE624"

# The Aria coordinate system is different than the Blender/NerfStudio coordinate system.
# Blender / Nerfstudio: +Z = back, +Y = up, +X = right
# Surreal: +Z = forward, +Y = down, +X = right
T_ARIA_NERFSTUDIO = SE3.from_matrix(
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
)


@dataclass
class AriaCameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_params: np.ndarray
    width: int
    height: int
    t_device_camera: SE3


@dataclass
class AriaImageFrame:
    camera: AriaCameraCalibration
    file_path: str
    t_world_camera: SE3
    timestamp_ns: float


@dataclass
class TimedPoses:
    timestamps_ns: np.ndarray
    t_world_devices: List[SE3]


def get_camera_calibs(provider: VrsDataProvider) -> Dict[str, AriaCameraCalibration]:
    """Retrieve the per-camera factory calibration from within the VRS."""

    factory_calib = {}
    name = "camera-rgb"
    device_calib = provider.get_device_calibration()
    assert device_calib is not None, "Could not find device calibration"
    sensor_calib = device_calib.get_camera_calib(name)
    assert sensor_calib is not None, f"Could not find sensor calibration for {name}"

    width = sensor_calib.get_image_size()[0].item()
    height = sensor_calib.get_image_size()[1].item()
    intrinsics = sensor_calib.projection_params()

    factory_calib[name] = AriaCameraCalibration(
        fx=intrinsics[0],
        fy=intrinsics[0],
        cx=intrinsics[1],
        cy=intrinsics[2],
        distortion_params=intrinsics[3:15],
        width=width,
        height=height,
        t_device_camera=sensor_calib.get_transform_device_camera(),
    )

    return factory_calib


def read_trajectory_csv_to_dict(file_iterable_csv: str) -> TimedPoses:
    closed_loop_traj = mps.read_closed_loop_trajectory(file_iterable_csv)  # type: ignore

    timestamps_secs, poses = zip(
        *[(it.tracking_timestamp.total_seconds(), it.transform_world_device) for it in closed_loop_traj]
    )

    SEC_TO_NANOSEC = 1e9
    return TimedPoses(
        timestamps_ns=(np.array(timestamps_secs) * SEC_TO_NANOSEC).astype(int),
        t_world_devices=poses,
    )


def to_aria_image_frame(
    provider: VrsDataProvider,
    index: int,
    name_to_camera: Dict[str, AriaCameraCalibration],
    t_world_devices: TimedPoses,
    output_dir: Path,
) -> AriaImageFrame:
    name = "camera-rgb"

    camera_calibration = name_to_camera[name]
    stream_id = provider.get_stream_id_from_label(name)
    assert stream_id is not None, f"Could not find stream {name}"

    # Get the image corresponding to this index
    image_data = provider.get_image_data_by_index(stream_id, index)
    img = Image.fromarray(image_data[0].to_numpy_array())
    capture_time_ns = image_data[1].capture_timestamp_ns

    file_path = f"{output_dir}/{name}_{capture_time_ns}.jpg"
    threading.Thread(target=lambda: img.save(file_path)).start()

    # Find the nearest neighbor pose with the closest timestamp to the capture time.
    nearest_pose_idx = np.searchsorted(t_world_devices.timestamps_ns, capture_time_ns)
    nearest_pose_idx = np.minimum(nearest_pose_idx, len(t_world_devices.timestamps_ns) - 1)
    assert nearest_pose_idx != -1, f"Could not find pose for {capture_time_ns}"
    t_world_device = t_world_devices.t_world_devices[nearest_pose_idx]

    # Compute the world to camera transform.
    t_world_camera = t_world_device @ camera_calibration.t_device_camera @ T_ARIA_NERFSTUDIO

    return AriaImageFrame(
        camera=camera_calibration,
        file_path=file_path,
        t_world_camera=t_world_camera,
        timestamp_ns=capture_time_ns,
    )


def to_nerfstudio_frame(frame: AriaImageFrame) -> Dict:
    return {
        "fl_x": frame.camera.fx,
        "fl_y": frame.camera.fy,
        "cx": frame.camera.cx,
        "cy": frame.camera.cy,
        "distortion_params": frame.camera.distortion_params.tolist(),
        "w": frame.camera.width,
        "h": frame.camera.height,
        "file_path": frame.file_path,
        "transform_matrix": frame.t_world_camera.to_matrix().tolist(),
        "timestamp": frame.timestamp_ns,
    }


@dataclass
class ProcessProjectAria:
    """Processes Project Aria data i.e. a VRS of the raw recording streams and the MPS attachments
    that provide poses, calibration, and 3d points. More information on MPS data can be found at:
      https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps.
    """

    vrs_file: Path
    """Path to the VRS file."""
    mps_data_dir: Path
    """Path to Project Aria Machine Perception Services (MPS) attachments."""
    output_dir: Path
    """Path to the output directory."""

    def main(self) -> None:
        """Generate a nerfstudio dataset from ProjectAria data (VRS) and MPS attachments."""
        # Create output directory if it doesn't exist.
        self.output_dir = self.output_dir.absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        provider = create_vrs_data_provider(str(self.vrs_file.absolute()))
        assert provider is not None, "Cannot open file"

        name_to_camera = get_camera_calibs(provider)

        print("Getting poses from closed loop trajectory CSV...")
        trajectory_csv = self.mps_data_dir / "closed_loop_trajectory.csv"
        t_world_devices = read_trajectory_csv_to_dict(str(trajectory_csv.absolute()))

        name = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(name)

        # create an AriaImageFrame for each image in the VRS.
        print("Creating Aria frames...")
        aria_frames = [
            to_aria_image_frame(provider, index, name_to_camera, t_world_devices, self.output_dir)
            for index in range(0, provider.get_num_data(stream_id))
        ]

        # create the NerfStudio frames from the AriaImageFrames.
        print("Creating NerfStudio frames...")
        CANONICAL_RGB_VALID_RADIUS = 707.5
        CANONICAL_RGB_WIDTH = 1408
        rgb_valid_radius = CANONICAL_RGB_VALID_RADIUS * (aria_frames[0].camera.width / CANONICAL_RGB_WIDTH)
        nerfstudio_frames = {
            "camera_model": ARIA_CAMERA_MODEL,
            "frames": [to_nerfstudio_frame(frame) for frame in aria_frames],
            "fisheye_crop_radius": rgb_valid_radius,
        }

        # save global point cloud, which is useful for Gaussian Splatting.
        points_path = self.mps_data_dir / "global_points.csv.gz"
        if not points_path.exists():
            # MPS point cloud output was renamed in Aria's December 4th, 2023 update.
            # https://facebookresearch.github.io/projectaria_tools/docs/ARK/sw_release_notes#project-aria-updates-aria-mobile-app-v140-and-changes-to-mps
            points_path = self.mps_data_dir / "semidense_points.csv.gz"

        if points_path.exists():
            print("Found global points, saving to PLY...")
            points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore
            points_data = filter_points_from_confidence(points_data)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array([cast(Any, it).position_world for it in points_data]))
            ply_file_path = self.output_dir / "global_points.ply"
            o3d.io.write_point_cloud(str(ply_file_path), pcd)

            nerfstudio_frames["ply_file_path"] = "global_points.ply"
        else:
            print("No global points found!")

        # write the json out to disk as transforms.json
        print("Writing transforms.json")
        transform_file = self.output_dir / "transforms.json"
        with open(transform_file, "w", encoding="UTF-8"):
            transform_file.write_text(json.dumps(nerfstudio_frames))


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessProjectAria).main()
