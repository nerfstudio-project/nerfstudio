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
    from projectaria_tools.core import mps, calibration
    from projectaria_tools.core.image import InterpolationMethod
    from projectaria_tools.core.data_provider import VrsDataProvider, create_vrs_data_provider
    from projectaria_tools.core.mps.utils import filter_points_from_confidence, get_nearest_pose
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
    pinhole_intrinsic: List[float]


@dataclass
class TimedPoses:
    timestamps_ns: np.ndarray
    t_world_devices: List[SE3]
    closed_loop_traj: List # It's a list of ClosedLoopTrajectoryPose objects, which is a C++ struct


def get_camera_calibs(provider: VrsDataProvider, name="camera-rgb") -> AriaCameraCalibration:
    """Retrieve the per-camera factory calibration from within the VRS."""

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
    closed_loop_traj = mps.read_closed_loop_trajectory(file_iterable_csv)  # has length 32689, so does timestamps_secs and poses.
    timestamps_secs, poses = zip(
        *[(it.tracking_timestamp.total_seconds(), it.transform_world_device) for it in closed_loop_traj]
    )
    SEC_TO_NANOSEC = 1e9
    # print(timestamps_secs[:10]) # prints (405.218581, 405.219225, 405.220223, 405.221243, 405.222218, 405.223262, 405.224214, 405.225212, 405.226213, 405.227208)
    return TimedPoses(
        timestamps_ns=(np.array(timestamps_secs) * SEC_TO_NANOSEC).astype(int),
        t_world_devices=poses,
        closed_loop_traj=closed_loop_traj,
    )

def undistort(provider: VrsDataProvider, sensor_name: str, index: int):# -> List[np.ndarray, tuple]:
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    image_data = provider.get_image_data_by_index(sensor_stream_id, index)
    image_array = image_data[0].to_numpy_array()

    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)
    f_length = src_calib.get_focal_lengths()[0].item()
    num_rows, num_cols = image_array.shape[0], image_array.shape[1]
    dst_calib = calibration.get_linear_camera_calibration(num_cols, num_rows, f_length, sensor_name)
    
    rectified_image = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
    """The linear camera model (a.k.a pinhole model) is parametrized by 4 coefficients : f_x, f_y, c_x, c_y."""
    # return None, None # UNCOMMENTING THIS FIXES THE SEGFAULT
    intrinsic = [f_length, f_length, num_cols/2, num_rows/2]

    return rectified_image, intrinsic

def to_aria_image_frame(
    provider: VrsDataProvider,
    index: int,
    name_to_camera: Dict[str, AriaCameraCalibration],
    t_world_devices: TimedPoses,
    output_dir: Path,
    name: str = "camera-rgb",

) -> AriaImageFrame:
    ari_camera_calibration = name_to_camera[name]
    stream_id = provider.get_stream_id_from_label(name)
    assert stream_id is not None, f"Could not find stream {name}"

    # Get the image corresponding to this index
    # ANTONIO NEW RECTIFYING BRANCH
    image_data = provider.get_image_data_by_index(stream_id, index)
    rectified_img, intrinsic = undistort(provider, name, index)
    #rectified_img = image_data[0].to_numpy_array()
    if len(rectified_img.shape) == 13: ##HEHEHE
        rectified_img = np.mean(rectified_img, axis=2).astype(np.uint8)
    img = Image.fromarray(rectified_img)
    capture_time_ns = image_data[1].capture_timestamp_ns
    
    
    # ANTONIO OLD BRANCH
    # image_data = provider.get_image_data_by_index(stream_id, index)
    # temp = image_data[0].to_numpy_array()
    # if len(temp.shape) == 3:
    #     img_np = np.mean(temp, axis=2).astype(np.uint8)
    #     img = Image.fromarray(img_np)
    # else:
    #     img = Image.fromarray(image_data[0].to_numpy_array())
    # capture_time_ns = image_data[1].capture_timestamp_ns
    

    
    file_path = f"{output_dir}/{name}_{capture_time_ns}.jpg"
    threading.Thread(target=lambda: img.save(file_path)).start()
    # print(f"{name}_{capture_time_ns}.jpg", img, name, index)
    
    # ANTONIO CALIBRATION-FACTO
    time_start = int(t_world_devices.closed_loop_traj[0].tracking_timestamp.total_seconds() * 1e9)
    time_end = int(t_world_devices.closed_loop_traj[-1].tracking_timestamp.total_seconds() * 1e9)

    capture_time_ns = np.clip(capture_time_ns, time_start + 1, time_end - 1)
    pose_info = get_nearest_pose(t_world_devices.closed_loop_traj, capture_time_ns)
    t_world_device = pose_info.transform_world_device
    print("Image Capture Time: ", capture_time_ns)
    print("Pose Capture Time: ", pose_info.tracking_timestamp.total_seconds() * 1e9)
    print("Difference: ", (capture_time_ns - pose_info.tracking_timestamp.total_seconds() * 1e9) / 1e9)
    print()
    
    # camera_stream_label = provider.get_label_from_stream_id() # should be the same as {name}
    device_calibration = provider.get_device_calibration()
    camera_calibration = device_calibration.get_camera_calib(name)
    
    T_device_camera = camera_calibration.get_transform_device_camera()
    t_world_camera = t_world_device @ T_device_camera @ T_ARIA_NERFSTUDIO # extrinsic matrix

    return AriaImageFrame(
        camera=ari_camera_calibration,
        file_path=file_path,
        t_world_camera=t_world_camera,
        timestamp_ns=capture_time_ns.item(),
        pinhole_intrinsic=intrinsic
    )


def to_nerfstudio_frame(frame: AriaImageFrame, pinhole: bool=False) -> Dict:
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
        
        names = ["camera-rgb", "camera-slam-left", "camera-slam-right"]
        name_to_camera = {name: get_camera_calibs(provider, name) for name in names} # name_to_camera is of type Dict[str, AriaCameraCalibration]
        
        print("Getting poses from closed loop trajectory CSV...")
        trajectory_csv = self.mps_data_dir / "closed_loop_trajectory.csv"
        t_world_devices = read_trajectory_csv_to_dict(str(trajectory_csv.absolute()))

        stream_ids = [provider.get_stream_id_from_label(name) for name in names] # prints [214-1, 1201-1, 1201-2], which is correct

        # create an AriaImageFrame for each image in the VRS.
        print("Creating Aria frames...")
        # aria_frames = [
        #     to_aria_image_frame(provider, index, name_to_camera, t_world_devices, self.output_dir)
        #     for index in range(0, provider.get_num_data(provider.get_stream_id_from_label("camera-rgb")))
        # ]
        aria_camera_frames = [
            [
                to_aria_image_frame(provider, index, name_to_camera, t_world_devices, self.output_dir, name=names[i])
                for index in range(0, provider.get_num_data(stream_id)) # there are 333 images per camera
                # for index in range(0, 1)
            ] 
            for i, stream_id in enumerate(stream_ids)
        ] # aria_frames = aria_camera_frames[0]
        all_aria_camera_frames = [to_nerfstudio_frame(frame) for camera_frames in aria_camera_frames for frame in camera_frames]

        
        # create the NerfStudio frames from the AriaImageFrames.
        print("Creating NerfStudio frames...")
        CANONICAL_RGB_VALID_RADIUS = 707.5
        CANONICAL_RGB_WIDTH = 1408
        rgb_valid_radius = CANONICAL_RGB_VALID_RADIUS * (aria_camera_frames[0][0].camera.width / CANONICAL_RGB_WIDTH) 
        
        # found here https://github.com/facebookresearch/projectaria_tools/blob/4aee633cb667ab927825dc10477cad0df8393a34/core/calibration/loader/SensorCalibrationJson.cpp#L102C5-L104C18 and divided by 2
        CANONICAL_SLAM_VALID_RADIUS = 165
        CANONICAL_SLAM_WIDTH = 640
        slam_valid_radius = 320.0 # CANONICAL_SLAM_VALID_RADIUS * (aria_camera_frames[1][0].camera.width / CANONICAL_SLAM_WIDTH) # equal to 165.0 in the end
        valid_radii = [rgb_valid_radius, slam_valid_radius, slam_valid_radius]
        
        # nerfstudio_frames = { # (this is the OG one from master/main branch)
        #     "camera_model": ARIA_CAMERA_MODEL,
        #     "frames": [to_nerfstudio_frame(frame) for frame in aria_frames],
        #     "fisheye_crop_radius": rgb_valid_radius,
        # }
        mainRGB_frames = { # same as OG nerfstudio_frames
            "camera_model": ARIA_CAMERA_MODEL,
            "frames": [to_nerfstudio_frame(frame) for frame in aria_camera_frames[0]],
            "fisheye_crop_radius": rgb_valid_radius, # if you remove this, the black corners appear
        }
        left_camera_frames = {
            "camera_model": ARIA_CAMERA_MODEL,
            "frames": [to_nerfstudio_frame(frame) for frame in aria_camera_frames[1]],
            "fisheye_crop_radius": slam_valid_radius,
        }
        right_camera_frames = {
            "camera_model": ARIA_CAMERA_MODEL,
            "frames": [to_nerfstudio_frame(frame) for frame in aria_camera_frames[2]],
            "fisheye_crop_radius": slam_valid_radius,
        }
        side_camera_frames = {
            "camera_model": ARIA_CAMERA_MODEL,
            "frames": left_camera_frames["frames"] + right_camera_frames["frames"],
            "fisheye_crop_radius": slam_valid_radius,
        }
        mainRGB_left_camera_frames = {
            "camera_model": ARIA_CAMERA_MODEL,
            "frames": mainRGB_frames["frames"] + left_camera_frames["frames"],
            "fisheye_crop_radius": slam_valid_radius,
        }
        from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
        # print("HELLO", CAMERA_MODELS["perspective"].value) # prints "OPENCV"
        all_cameras_grayscale_frames = { # if you want to use this, make sure do turn on the HEHEHE change to ==3 instead of ==13
            "camera_model": CAMERA_MODELS["perspective"].value,
            "frames": all_aria_camera_frames,
            # "fisheye_crop_radius": slam_valid_radius,
        }

        # IT'S PINHOLE TIME
        mainRGB_pinhole_frames = {
            "camera_model": CAMERA_MODELS["perspective"].value,
            "frames": [to_nerfstudio_frame(frame, pinhole=True) for frame in aria_camera_frames[0]],
        }
        left_pinhole_frames = {
            "camera_model": CAMERA_MODELS["perspective"].value,
            "frames": [to_nerfstudio_frame(frame, pinhole=True) for frame in aria_camera_frames[1]],
        }
        right_pinhole_frames = {
            "camera_model": CAMERA_MODELS["perspective"].value,
            "frames": [to_nerfstudio_frame(frame, pinhole=True) for frame in aria_camera_frames[2]],
        }
        side_camera_pinhole_frames = {
            "camera_model": CAMERA_MODELS["perspective"].value,
            "frames": left_pinhole_frames["frames"] + right_pinhole_frames["frames"],
        }
        all_cameras_grayscale_pinhole_frames = { # if you want to use this, make sure do turn on the HEHEHE change to ==3 instead of ==13
            "camera_model": CAMERA_MODELS["perspective"].value,
            "frames": mainRGB_pinhole_frames["frames"] + side_camera_pinhole_frames["frames"],
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
            mainRGB_frames["ply_file_path"] = "global_points.ply"
        else:
            print("No global points found!")

        # write the json out to disk as transforms.json
        print("Writing transforms.json")
        transform_file = self.output_dir / "transforms.json"
        with open(transform_file, "w", encoding="UTF-8"):
            # transform_file.write_text(json.dumps(mainRGB_pinhole_frames))
            # transform_file.write_text(json.dumps(side_camera_frames))
            # transform_file.write_text(json.dumps(left_pinhole_frames))
            # transform_file.write_text(json.dumps(right_pinhole_frames))
            # transform_file.write_text(json.dumps(side_camera_pinhole_frames))
            transform_file.write_text(json.dumps(all_cameras_grayscale_pinhole_frames)) # make sure to change HEHEHE 13
        del provider
        import importlib.metadata
        print(importlib.metadata.version("projectaria_tools"))

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessProjectAria).main()