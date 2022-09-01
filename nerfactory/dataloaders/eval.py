# Copyright 2022 The Plenoptix Team. All rights reserved.
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
Data loader.
"""

import random
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch

from nerfactory.cameras.cameras import Cameras
from nerfactory.cameras.rays import RayBundle
from nerfactory.dataloaders.image_dataset import ImageDataset
from nerfactory.utils.misc import get_dict_to_torch


class EvalDataloader:
    """Evaluation dataloader base class

    Args:
        image_dataset: ImageDataset to load data from
        cameras: Cameras to use for generating rays
        num_rays_per_chunk: Number of camera rays to generate per chunk
        device: Device to load data to
    """

    def __init__(
        self,
        image_dataset: ImageDataset,
        cameras: Cameras,
        num_rays_per_chunk: int,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.image_dataset = image_dataset
        self.cameras = cameras
        self.num_rays_per_chunk = num_rays_per_chunk
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Cameras:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        distortion_params = None
        if self.cameras.distortion_params is not None:
            distortion_params = self.cameras.distortion_params[image_idx]

        camera = Cameras(
            fx=self.cameras.fx[image_idx],
            fy=self.cameras.fy[image_idx],
            cx=self.cameras.cx,
            cy=self.cameras.cy,
            camera_to_worlds=self.cameras.camera_to_worlds[image_idx],
            distortion_params=distortion_params,
            camera_type=self.cameras.camera_type,
        )
        return camera

    def get_data_from_image_idx(self, image_idx) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index."""
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx)
        ray_bundle.num_rays_per_chunk = self.num_rays_per_chunk
        ray_bundle.camera_indices = torch.Tensor([image_idx])[..., None].int()
        batch = self.image_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device)
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices."""

    def __init__(
        self,
        image_dataset: ImageDataset,
        cameras: Cameras,
        num_rays_per_chunk: int,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        """
        Args:
            image_dataset: ImageDataset to load data from
            cameras: Cameras to use for generating rays
            num_rays_per_chunk: Number of camera rays to generate per chunk
            image_indices: List of image indices to load data from. If None, then use all images.
            device: Device to load data to
        """
        super().__init__(image_dataset, cameras, num_rays_per_chunk, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(image_dataset)))
        else:
            self.image_indices = image_indices
        self.num_rays_per_chunk = num_rays_per_chunk
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            ray_bundle.num_rays_per_chunk = self.num_rays_per_chunk
            self.count += 1
            return ray_bundle, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.

    Args:
        image_dataset: ImageDataset to load data from
        cameras: Cameras to use for generating rays
        num_rays_per_chunk: Number of camera rays to generate per chunk
        device: Device to load data to
    """

    def __init__(
        self,
        image_dataset: ImageDataset,
        cameras: Cameras,
        num_rays_per_chunk: int,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(image_dataset, cameras, num_rays_per_chunk, device, **kwargs)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < 1:
            image_indices = range(self.cameras.size)
            image_idx = random.choice(image_indices)
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        raise StopIteration
