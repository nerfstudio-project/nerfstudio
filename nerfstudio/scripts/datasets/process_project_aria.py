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
import random
import sys
import threading
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast

import numpy as np
import open3d as o3d
import tyro
from PIL import Image

try:
    from projectaria_tools.core import calibration, mps
    from projectaria_tools.core.data_provider import VrsDataProvider, create_vrs_data_provider
    from projectaria_tools.core.image import InterpolationMethod
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
    pinhole_intrinsic: Tuple[float, float, float, float]


@dataclass
class TimedPoses:
    timestamps_ns: np.ndarray
    t_world_devices: List[SE3]


def get_camera_calibs(
    provider: VrsDataProvider, name: Literal["camera-rgb", "camera-slam-left", "camera-slam-right"] = "camera-rgb"
) -> AriaCameraCalibration:
    """Retrieve the per-camera factory calibration from within the VRS."""
    assert name in ["camera-rgb", "camera-slam-left", "camera-slam-right"], f"{name} is not a valid camera sensor"
    factory_calib = {}
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

    return factory_calib[name]


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


def undistort_image_and_calibration(
    input_image: np.ndarray,
    input_calib: calibration.CameraCalibration,
    output_focal_length: int,
) -> Tuple[np.ndarray, calibration.CameraCalibration]:
    """
    Return the undistorted image and the updated camera calibration.
    """
    input_calib_width = input_calib.get_image_size()[0]
    input_calib_height = input_calib.get_image_size()[1]
    if input_image.shape[1] != input_calib_width or input_image.shape[0] != input_calib_height:
        raise ValueError(
            f"Input image shape {input_image.shape} does not match calibration {input_calib.get_image_size()}"
        )

    # Undistort the image
    pinhole_calib = calibration.get_linear_camera_calibration(
        int(input_calib_width),
        int(input_calib_height),
        output_focal_length,
        "pinhole",
        input_calib.get_transform_device_camera(),
    )
    output_image = calibration.distort_by_calibration(
        input_image, pinhole_calib, input_calib, InterpolationMethod.BILINEAR
    )

    return output_image, pinhole_calib


def rotate_upright_image_and_calibration(
    input_image: np.ndarray,
    input_calib: calibration.CameraCalibration,
) -> Tuple[np.ndarray, calibration.CameraCalibration]:
    """
    Return the rotated upright image and update both intrinsics and extrinsics of the camera calibration
    NOTE: This function only supports pinhole and fisheye624 camera model.
    """
    output_image = np.rot90(input_image, k=3)
    updated_calib = calibration.rotate_camera_calib_cw90deg(input_calib)

    return output_image, updated_calib


def generate_circular_mask(numRows: int, numCols: int, radius: float) -> np.ndarray:
    """
    Generates a mask where a circle in the center of the image with input radius is white (sampled from).
    Everything outside the circle is black (masked out)
    """
    # Calculate the center coordinates
    rows, cols = np.ogrid[:numRows, :numCols]
    center_row, center_col = numRows // 2, numCols // 2

    # Calculate the distance of each pixel from the center
    distance_from_center = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    mask = np.zeros((numRows, numCols), dtype=np.uint8)
    mask[distance_from_center <= radius] = 1
    return mask


def to_aria_image_frame(
    provider: VrsDataProvider,
    index: int,
    name_to_camera: Dict[str, AriaCameraCalibration],
    t_world_devices: TimedPoses,
    output_dir: Path,
    camera_name: str = "camera-rgb",
    pinhole: bool = False,
) -> AriaImageFrame:
    aria_cam_calib = name_to_camera[camera_name]
    stream_id = provider.get_stream_id_from_label(camera_name)
    assert stream_id is not None, f"Could not find stream {camera_name}"

    # Retrieve the current camera calibration
    device_calib = provider.get_device_calibration()
    assert device_calib is not None, "Could not find device calibration"
    src_calib = device_calib.get_camera_calib(camera_name)
    assert isinstance(src_calib, calibration.CameraCalibration), "src_calib is not of type CameraCalibration"

    # Get the image corresponding to this index and undistort it
    image_data = provider.get_image_data_by_index(stream_id, index)
    image_array, intrinsic = image_data[0].to_numpy_array().astype(np.uint8), (0, 0, 0, 0)
    if pinhole:
        f_length = 500 if camera_name == "camera-rgb" else 170
        image_array, src_calib = undistort_image_and_calibration(image_array, src_calib, f_length)
        intrinsic = (f_length, f_length, image_array.shape[1] // 2, image_array.shape[0] // 2)

    # Rotate the image right side up
    image_array, src_calib = rotate_upright_image_and_calibration(image_array, src_calib)
    img = Image.fromarray(image_array)
    capture_time_ns = image_data[1].capture_timestamp_ns
    intrinsic = (intrinsic[0], intrinsic[1], intrinsic[3], intrinsic[2])

    # Save the image
    file_path = f"{output_dir}/{camera_name}_{capture_time_ns}.jpg"
    threading.Thread(target=lambda: img.save(file_path)).start()

    # Find the nearest neighbor pose with the closest timestamp to the capture time.
    nearest_pose_idx = np.searchsorted(t_world_devices.timestamps_ns, capture_time_ns)
    nearest_pose_idx = np.minimum(nearest_pose_idx, len(t_world_devices.timestamps_ns) - 1)
    assert nearest_pose_idx != -1, f"Could not find pose for {capture_time_ns}"
    t_world_device = t_world_devices.t_world_devices[nearest_pose_idx]

    # Compute the world to camera transform.
    t_world_camera = t_world_device @ src_calib.get_transform_device_camera() @ T_ARIA_NERFSTUDIO

    # Define new AriaCameraCalibration since we rotated the image to be upright
    width = src_calib.get_image_size()[0].item()
    height = src_calib.get_image_size()[1].item()
    intrinsics = src_calib.projection_params()
    aria_cam_calib = AriaCameraCalibration(
        fx=intrinsics[0],
        fy=intrinsics[0],
        cx=intrinsics[1],
        cy=intrinsics[2],
        distortion_params=intrinsics[3:15],
        width=width,
        height=height,
        t_device_camera=src_calib.get_transform_device_camera(),
    )

    return AriaImageFrame(
        camera=aria_cam_calib,
        file_path=file_path,
        t_world_camera=t_world_camera,
        timestamp_ns=capture_time_ns,
        pinhole_intrinsic=intrinsic,
    )


def to_nerfstudio_frame(frame: AriaImageFrame, pinhole: bool = False, mask_path: str = "") -> Dict:
    if pinhole:
        return {
            "fl_x": frame.pinhole_intrinsic[0],
            "fl_y": frame.pinhole_intrinsic[1],
            "cx": frame.pinhole_intrinsic[2],
            "cy": frame.pinhole_intrinsic[3],
            "w": frame.pinhole_intrinsic[2] * 2,
            "h": frame.pinhole_intrinsic[3] * 2,
            "file_path": frame.file_path,
            "transform_matrix": frame.t_world_camera.to_matrix().tolist(),
            "timestamp": frame.timestamp_ns,
            "mask_path": mask_path,
        }
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

    vrs_file: Tuple[Path, ...]
    """Path to the VRS file(s)."""
    mps_data_dir: Tuple[Path, ...]
    """Path to Project Aria Machine Perception Services (MPS) attachments."""
    output_dir: Path
    """Path to the output directory."""
    points_file: Tuple[Path, ...] = ()
    """Path to the point cloud file (usually called semidense_points.csv.gz) if not in the mps_data_dir"""
    include_side_cameras: bool = False
    """If True, include and process the images captured by the grayscale side cameras. 
    If False, only uses the main RGB camera's data."""
    max_dataset_size: int = -1
    """Max number of images to train on. If the provided vrs_file has more images than max_dataset_size, 
    images will be sampled approximately evenly. If max_dataset_size=-1, use all images available."""

    def main(self) -> None:
        """Generate a nerfstudio dataset from ProjectAria data (VRS) and MPS attachments."""
        # Create output directory if it doesn't exist
        self.output_dir = self.output_dir.absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create list of tuples containing files from each wearer and output variables
        assert len(self.vrs_file) == len(
            self.mps_data_dir
        ), "Please provide an Aria MPS attachment for each corresponding VRS file."
        vrs_mps_points_triplets = list(zip_longest(self.vrs_file, self.mps_data_dir, self.points_file))  # type: ignore
        num_recordings = len(vrs_mps_points_triplets)
        nerfstudio_frames = {
            "camera_model": "OPENCV" if self.include_side_cameras else ARIA_CAMERA_MODEL,
            "frames": [],
        }
        points = []

        # Process the aria data of each user one by one
        for rec_i, (vrs_file, mps_data_dir, points_file) in enumerate(vrs_mps_points_triplets):
            provider = create_vrs_data_provider(str(vrs_file.absolute()))
            assert provider is not None, "Cannot open file"

            names = ["camera-rgb", "camera-slam-left", "camera-slam-right"]
            name_to_camera = {
                name: get_camera_calibs(provider, name)  # type: ignore
                for name in names
            }  # name_to_camera is of type Dict[str, AriaCameraCalibration]

            print(f"Getting poses from recording {rec_i + 1}'s closed loop trajectory CSV...")
            trajectory_csv = mps_data_dir / "closed_loop_trajectory.csv"
            t_world_devices = read_trajectory_csv_to_dict(str(trajectory_csv.absolute()))

            stream_ids = [provider.get_stream_id_from_label(name) for name in names]

            # Create an AriaImageFrame for each image in the VRS
            print(f"Creating Aria frames for recording {rec_i + 1}...")
            CANONICAL_RGB_VALID_RADIUS = 707.5  # radius of a circular mask that represents the valid area on the camera's sensor plane. Pixels out of this circular region are considered invalid
            CANONICAL_RGB_WIDTH = 1408
            total_num_images_per_camera = provider.get_num_data(stream_ids[0])
            if self.max_dataset_size == -1:
                num_images_to_sample_per_camera = total_num_images_per_camera
            else:
                num_images_to_sample_per_camera = (
                    self.max_dataset_size // (len(vrs_mps_points_triplets) * 3)
                    if self.include_side_cameras
                    else self.max_dataset_size // len(vrs_mps_points_triplets)
                )
            sampling_indicies = random.sample(range(total_num_images_per_camera), num_images_to_sample_per_camera)
            if not self.include_side_cameras:
                aria_rgb_frames = [
                    to_aria_image_frame(
                        provider, index, name_to_camera, t_world_devices, self.output_dir, camera_name=names[0]
                    )
                    for index in sampling_indicies
                ]
                print(f"Creating NerfStudio frames for recording {rec_i + 1}...")
                nerfstudio_frames["frames"] += [to_nerfstudio_frame(frame) for frame in aria_rgb_frames]
                rgb_valid_radius = CANONICAL_RGB_VALID_RADIUS * (
                    aria_rgb_frames[0].camera.width / CANONICAL_RGB_WIDTH
                )  # to handle both high-res 2880 x 2880 aria captures
                nerfstudio_frames["fisheye_crop_radius"] = rgb_valid_radius
            else:
                aria_all3cameras_pinhole_frames = [
                    [
                        to_aria_image_frame(
                            provider,
                            index,
                            name_to_camera,
                            t_world_devices,
                            self.output_dir,
                            camera_name=names[i],
                            pinhole=True,
                        )
                        for index in range(provider.get_num_data(stream_id))
                    ]
                    for i, stream_id in enumerate(stream_ids)
                ]
                # Generate masks for undistorted images
                rgb_width = aria_all3cameras_pinhole_frames[0][0].camera.width
                rgb_valid_radius = CANONICAL_RGB_VALID_RADIUS * (rgb_width / CANONICAL_RGB_WIDTH)
                slam_valid_radius = 330.0  # found here: https://github.com/facebookresearch/projectaria_tools/blob/4aee633cb667ab927825dc10477cad0df8393a34/core/calibration/loader/SensorCalibrationJson.cpp#L102C5-L104C18
                rgb_mask_nparray, slam_mask_nparray = (
                    generate_circular_mask(rgb_width, rgb_width, rgb_valid_radius),
                    generate_circular_mask(640, 480, slam_valid_radius),
                )
                rgb_mask_filepath, slam_mask_filepath = (
                    f"{self.output_dir}/rgb_mask.jpg",
                    f"{self.output_dir}/slam_mask.jpg",
                )
                Image.fromarray(rgb_mask_nparray).save(rgb_mask_filepath)
                Image.fromarray(slam_mask_nparray).save(slam_mask_filepath)

                print(f"Creating NerfStudio frames for recording {rec_i + 1}...")
                mask_filepaths = [rgb_mask_filepath, slam_mask_filepath, slam_mask_filepath]
                pinhole_frames = [
                    to_nerfstudio_frame(frame, pinhole=True, mask_path=mask_filepath)
                    for i, mask_filepath in enumerate(mask_filepaths)
                    for frame in aria_all3cameras_pinhole_frames[i]
                ]
                nerfstudio_frames["frames"] += pinhole_frames

            if points_file is not None:
                points_path = points_file
            else:
                points_path = mps_data_dir / "global_points.csv.gz"
                if not points_path.exists():
                    # MPS point cloud output was renamed in Aria's December 4th, 2023 update.
                    # https://facebookresearch.github.io/projectaria_tools/docs/ARK/sw_release_notes#project-aria-updates-aria-mobile-app-v140-and-changes-to-mps
                    points_path = mps_data_dir / "semidense_points.csv.gz"

            if points_path.exists():
                print(f"Found global points for recording {rec_i+1}")
                points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore
                points_data = filter_points_from_confidence(points_data)
                points += [cast(Any, it).position_world for it in points_data]
        print(len(nerfstudio_frames['frames']))
        if len(points) > 0:
            print("Saving found points to PLY...")
            print(f"Total number of points found: {len(points)} in {num_recordings} recording(s) provided")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            ply_file_path = self.output_dir / "global_points.ply"
            o3d.io.write_point_cloud(str(ply_file_path), pcd)
            nerfstudio_frames["ply_file_path"] = "global_points.ply"
        else:
            print("No global points found!")

        # Write the json out to disk as transforms.json
        print("Writing transforms.json")
        transform_file = self.output_dir / "transforms.json"
        with open(transform_file, "w", encoding="UTF-8"):
            transform_file.write_text(json.dumps(nerfstudio_frames))


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessProjectAria).main()
