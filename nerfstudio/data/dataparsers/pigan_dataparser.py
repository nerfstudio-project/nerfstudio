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

"""Data parser for PI-GAN

# https://arxiv.org/pdf/2012.00926
"""

from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Type

import numpy as np
import torch
import math
import imageio

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class PiganDataParserConfig(DataParserConfig):
    """FFHQ dataset config"""

    _target: Type = field(default_factory=lambda: PiganDataParser)
    """target class to instantiate"""
    data: Path = Path("data/FFHQ")
    """Directory specifying location of data""" 
    data_type: str = 'thumbnail'  
    """data_type of FFHQ ( thumbnail or image or in the wild ...)"""
    fov : float = 12.
    """field of view determined by dataset manually"""
    radius : float = 1.
    """sample random camera pose by pose sample mode distribution"""
    pose_sample_mode : str = 'noraml'


@dataclass
class PiganDataParser(DataParser):
    """PiGAN Dataset"""

    config: PiganDataParserConfig

    def __init__(self, config: PiganDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "ffhq-dataset-v2.json")
            data_dir = self.config.data

    def _generate_dataparser_outputs(self, split="train"): 
        # FIXME
        # meta = load_from_json(self.config.data / f"ffhq-dataset-v2-{split}.json")
        meta = load_from_json(self.config.data / f"ffhq-dataset-v2.json")
        data_dir = self.config.data
        
        image_filenames = []
        poses = []

        for num in meta.values():
            filepath = Path(num[self.config.data_type]['file_path'].split('/')[-1]) 
            fname = data_dir / filepath #ANCHOR 
            image_filenames.append(fname)

        # NOTE - EG3D올리기 전까지는 굳이 real image의 camera pose를 넘겨 discriminator에 정보로 줄 필요는 없어보인다. 만약 필요해지면 아예 다시 구성해야한다. => 아예 dataparser에 cameras가 아닌 다른 argument에 camera pose원본들을 넘겨서 batch에 추가해서 discriminator에서 사용하도록 해주면 될 듯
        # NOTE - if poses givne (like eg3d), add 'transform matrix' to json
        #     poses.append(np.array(frame["transform_matrix"]))
        # poses = np.array(poses).astype(np.float32)

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        self.num_rays_per_image = image_height * image_width

        # calculate camera intrinsic parameters
        camera_angle = 2 * math.pi * self.config.fov / 360
        focal_length =  0.5 * image_width / np.tan(0.5 * camera_angle )

        cx = image_width / 2.0
        cy = image_height / 2.0

        # ranodm camera pose generation
        # REVIEW - pigan은 random camera pose들을 생성, 일정 공간에서 몇개나 random camera pose들을 세팅할 것인지 = num_cameras
        num_cameras = 70000 # 데이터 분포 -> index로 설정될 거임. 따라서 최소 데이터개수 이상을 사용해야함.
        camera_to_world = self.sample_camera_transform_matrix(num_cameras)  # num_cameras가 dataloader에서 self.cameras.size

        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world, 
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )
        print(camera_to_world.shape)
            

        # camera to world가 dictuonary여야함 ->지금 그냥 tensor


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )
     
        return dataparser_outputs


    def sample_camera_transform_matrix(self, n):
        """
        Return sampled random camera pose with transforming to cam2world matrix 
        """
        camera_origin, _, _ = self._sample_camera_positions(n=n, r=self.config.radius, mode=self.config.pose_sample_mode)
        forward_vector = self._normalize_vecs(-camera_origin)
        camera_transform_matrix = self._create_cam2world_matrix(forward_vector, camera_origin)
        return camera_transform_matrix

    # NOTE - world coordinate positions sampling
    # https://github.com/marcoamonteiro/pi-GAN
    def _sample_camera_positions(self, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
        """
        Samples n random locations along a sphere of radius r. Uses the specified distribution.
        Theta is yaw in radians (-pi, pi)
        Phi is pitch in radians (0, pi)
        """

        if mode == 'uniform':
            theta = (torch.rand((n, 1)) - 0.5) * 2 * horizontal_stddev + horizontal_mean
            phi = (torch.rand((n, 1)) - 0.5) * 2 * vertical_stddev + vertical_mean

        elif mode == 'normal' or mode == 'gaussian':
            theta = torch.randn((n, 1)) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1)) * vertical_stddev + vertical_mean

        elif mode == 'hybrid':
            if random.random() < 0.5:
                theta = (torch.rand((n, 1)) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
                phi = (torch.rand((n, 1)) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
            else:
                theta = torch.randn((n, 1)) * horizontal_stddev + horizontal_mean
                phi = torch.randn((n, 1)) * vertical_stddev + vertical_mean

        elif mode == 'spherical_uniform':
            theta = (torch.rand((n, 1)) - .5) * 2 * horizontal_stddev + horizontal_mean
            v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
            v = ((torch.rand((n,1)) - .5) * 2 * v_stddev + v_mean)
            v = torch.clamp(v, 1e-5, 1 - 1e-5)
            phi = torch.arccos(1 - 2 * v)

        else:
            # Just use the mean.
            theta = torch.ones((n, 1), dtype=torch.float) * horizontal_mean
            phi = torch.ones((n, 1), dtype=torch.float) * vertical_mean

        phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

        output_points = torch.zeros((n, 3))
        output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
        output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
        output_points[:, 1:2] = r*torch.cos(phi)

        return output_points, phi, theta

    def _create_cam2world_matrix(self, forward_vector, origin):
        """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

        #NOTE - world coordinate에서는 camera origing to 0,0,0가 forward vector일 것임. 

        # world up_vector
        up_vector = torch.tensor([0, 1, 0], dtype=torch.float).expand_as(forward_vector)

        # 
        left_vector = self._normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

        up_vector = self._normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

        rotation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
        rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

        translation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
        translation_matrix[:, :3, 3] = origin

        cam2world = translation_matrix @ rotation_matrix
        cam2world = cam2world[:,:3,:]

        return cam2world

    def _normalize_vecs(self, vectors):
        """
        Normalize vector lengths.
        """
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
        
