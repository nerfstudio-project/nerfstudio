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

"""
Mock dataparser for sharing model checkpoints without requiring original training data.
Creates dummy camera poses and image paths for inference/viewing purposes only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch
import numpy as np

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


@dataclass
class MockDataParserConfig(DataParserConfig):
    """Mock dataset config for inference without original data"""

    _target: Type = field(default_factory=lambda: MockDataParser)
    """target class to instantiate"""
    num_cameras: int = 100
    """number of dummy cameras to generate"""
    image_height: int = 800
    """height of dummy images"""
    image_width: int = 800
    """width of dummy images"""
    focal_length: float = 800.0
    """focal length for dummy cameras"""
    scene_scale: float = 1.0
    """scene scale"""


@dataclass
class MockDataParser(DataParser):
    """Mock DataParser that generates dummy data for inference/viewing without original training data"""

    config: MockDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        """Generate mock dataparser outputs with dummy camera poses and image paths"""
        
        # Generate dummy image filenames - these don't need to exist since we're only doing inference
        image_filenames = [Path(f"mock_image_{i:04d}.jpg") for i in range(self.config.num_cameras)]
        
        # Generate camera poses in a reasonable sphere around the scene
        poses = self._generate_spherical_poses(self.config.num_cameras)
        
        # Create camera intrinsics
        fx = fy = self.config.focal_length
        cx = self.config.image_width / 2.0
        cy = self.config.image_height / 2.0
        
        cameras = Cameras(
            fx=torch.full((self.config.num_cameras,), fx),
            fy=torch.full((self.config.num_cameras,), fy),
            cx=torch.full((self.config.num_cameras,), cx),
            cy=torch.full((self.config.num_cameras,), cy),
            height=torch.full((self.config.num_cameras,), self.config.image_height),
            width=torch.full((self.config.num_cameras,), self.config.image_width),
            camera_to_worlds=poses,
            camera_type=torch.full((self.config.num_cameras,), CameraType.PERSPECTIVE.value),
        )
        
        # Default scene box
        scene_box = SceneBox(aabb=torch.tensor([[-self.config.scene_scale, -self.config.scene_scale, -self.config.scene_scale], 
                                               [self.config.scene_scale, self.config.scene_scale, self.config.scene_scale]]))
        
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_transform=torch.eye(4)[:3, :],
            dataparser_scale=1.0,
        )
        
        return dataparser_outputs
    
    def _generate_spherical_poses(self, num_poses: int) -> torch.Tensor:
        """Generate camera poses distributed on a sphere looking at the origin"""
        poses = []
        
        # Generate poses on a sphere
        for i in range(num_poses):
            # Spherical coordinates
            theta = 2 * np.pi * i / num_poses  # azimuth
            phi = np.pi / 4  # elevation (45 degrees)
            radius = 4.0
            
            # Convert to Cartesian
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)  
            z = radius * np.cos(phi)
            
            # Look at origin
            camera_position = np.array([x, y, z])
            look_at = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])
            
            # Create camera-to-world matrix
            forward = look_at - camera_position
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            up_corrected = np.cross(right, forward)
            
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up_corrected
            pose[:3, 2] = -forward  # -forward for OpenCV convention
            pose[:3, 3] = camera_position
            
            poses.append(pose[:3, :4])
        
        return torch.from_numpy(np.stack(poses)).float()