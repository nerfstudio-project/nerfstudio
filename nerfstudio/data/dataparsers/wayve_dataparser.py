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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Literal, Optional, Type

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.poses import inverse, to4x4

console = Console()

MAX_AUTO_RESOLUTION = 1600

def get_image_mask(image_idx: int, index_to_camera_position, image_masks):
    camera_position = index_to_camera_position[image_idx]
    pil_mask = image_masks[camera_position]
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    return {"mask": mask_tensor}

def wayve_run_pose_to_nerfstudio_pose(P_run, mean_translation, scale_factor, G_nerf_run):
    P_nerf = G_nerf_run.to(P_run.device) @ P_run
    P_nerf[:, :3, 3] = (P_nerf[:, :3, 3] - mean_translation.to(P_nerf.device)) * scale_factor.to(P_nerf.device)
    transform1 = torch.tensor([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float, device=P_run.device)
    P_nerf = transform1 @ P_nerf

    # convert from opencv camera to nerfstudio camera
    P_nerf[:, 0:3, 1:3] *= -1
    P_nerf = P_nerf[:, np.array([1, 0, 2, 3]), :]
    P_nerf[:, 2, :] *= -1
    # rotate so z-up in nerfstudio viewer
    transform2 = torch.tensor([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float, device=P_run.device)
    P_nerf = transform2 @ P_nerf
    
    return P_nerf
@dataclass
class WayveDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: WayveDataParser)
    """target class to instantiate"""
    data: Path = Path("/mnt/remote/data/users/nikhil/2022-06-27--08-27-58--session_2022_06_25_03_10_41_host_zak_wayve_start-stop_pre-int_img-aug_4")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 2
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "none"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    start_timestamp_us: int = 1656318618168677
    end_timestamp_us: int  = 1656318649646730
    distance_threshold_between_frames_m: float = 10.0
    frame_rate: float = 25


@dataclass
class WayveDataParser(DataParser):
    """WayveData DatasetParser"""

    config: WayveDataParserConfig
    downscale_factor: Optional[int] = None


    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        root_dir = Path('/mnt/remote/image_storage/image-cache/jpeg-full_resolution/')
        run_id = Path('sedna/2022-06-27--08-27-58--session_2022_06_25_03_10_41_host_zak_wayve_start-stop_pre-int_img-aug_4')
        images_path = str(root_dir / run_id / "cameras")
        data = Path("/mnt/remote/data/users/nikhil/2022-06-27--08-27-58--session_2022_06_25_03_10_41_host_zak_wayve_start-stop_pre-int_img-aug_4")
        calibration = load_from_json(data/"calibration.json")
        # Uncomment when multicam support is ready
        # camera_positions = list(calibration.keys())
        camera_positions = ['front-forward']
        vehicle_poses = np.load(data/"egopose.npz")
        df = pd.read_parquet(data/"data.parquet")
        image_masks = {}
        for camera_position in camera_positions:
            mask = Image.open(data / f"masks/{camera_position}.png")
            if self.config.downscale_factor is not None:
                width, height = mask.size
                newsize = (int(width/self.config.downscale_factor), int(height/self.config.downscale_factor))
                mask = mask.resize(newsize)
            image_masks[camera_position] = mask
        run_global_egopose = vehicle_poses["egopose"].reshape(-1, 4, 4)
        nan_mask = np.sum(np.isnan(run_global_egopose), axis=(-1, -2)) == 0
        ff_timestamp = df['key__cameras__front_forward__image_timestamp_unixus'].to_numpy()
        timestamp_mask = (ff_timestamp > self.config.start_timestamp_us) & (ff_timestamp < self.config.end_timestamp_us)
        mask = nan_mask & timestamp_mask
        segment_global_egopose = run_global_egopose[mask]
        df = df[mask]
        speed = df['inferred__state__odometry__speed_kmh'].to_numpy(dtype=np.float32) / 3.6
        distance = np.cumsum(speed) / self.config.frame_rate
        ticks = np.arange(0, distance[-1], self.config.distance_threshold_between_frames_m)
        if split in ["test", "val"]:
            ticks = ticks[:-1] + self.config.distance_threshold_between_frames_m / 2
        indices = np.searchsorted(distance, ticks)
        if split == "render":
            indices = np.arange(len(distance))
        image_filenames = []
        wayve_poses = {}
        intrinsics = []
        distortion = []
        num_rows = len(df)
        # calibration = {'front-forward': calibration['front-forward']}
        image_index_to_mask = {}
        
        for camera_position in camera_positions:
            camera_calibration = calibration[camera_position]
            camera_str = camera_position.replace('-', '_')
            for index in range(len(image_filenames), len(image_filenames) + num_rows):
                image_index_to_mask[index] = camera_position
            image_ts_column = f'key__cameras__{camera_str}__image_timestamp_unixus'
            image_filenames.append(np.array([f'{images_path}/{camera_position}/{ts}unixus.jpeg' for ts in df[image_ts_column].to_list()]))
            camera_pose = np.array(camera_calibration['pose']).reshape(1, 4, 4)
            image_global_pose = segment_global_egopose @ camera_pose
            wayve_poses[camera_position] = torch.from_numpy(image_global_pose).float()
            intrinsics.append(torch.tensor(camera_calibration['intrinsics']).view(1, 3, 3).expand(num_rows, -1, -1))
            distortion.append(torch.tensor(camera_calibration['distortion'][:6]).view(1, 6).expand(num_rows, -1))
            
        # Move first frame to be at the origin
        self.G_nerf_run = to4x4(inverse(wayve_poses['front-forward'][:1])).squeeze(0)

        concat_wayve_poses =torch.stack([wayve_poses[pos] for pos in camera_positions], dim=1)
        segment_poses = self.G_nerf_run @ concat_wayve_poses
        translation = segment_poses[:, :, :3, 3]
        self.mean_translation = torch.mean(translation, dim=(0, 1))
        self.scale_factor = 1.0 / torch.max(torch.abs(translation - self.mean_translation))
        
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=1))
        distortion = torch.from_numpy(np.stack(distortion, axis=1))
        
        poses = concat_wayve_poses[indices].reshape(-1, 4, 4)
        intrinsics = intrinsics[indices].reshape(-1, 3, 3)
        distortion = distortion[indices].reshape(-1, 6)
        poses = wayve_run_pose_to_nerfstudio_pose(poses, self.mean_translation, self.scale_factor, self.G_nerf_run)
        
        image_filenames = np.stack(image_filenames, axis=1)
        image_filenames = image_filenames[indices].reshape(-1).tolist()
        
        camera_type = CameraType.FISHEYE

        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            distortion_params=distortion,
            height=1280,
            width=2048,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )
        if self.config.downscale_factor is not None:
            cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor) 
        aabb_scale = 1.0
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            image_scale_factor=1/self.config.downscale_factor if self.config.downscale_factor is not None else None,
            additional_inputs={"masks": {"func": get_image_mask, "kwargs": {"index_to_camera_position": image_index_to_mask, "image_masks": image_masks}}},
        )
        return dataparser_outputs

if __name__ == "__main__":
    data = WayveDataParser(config=WayveDataParserConfig())
    data._generate_dataparser_outputs()
