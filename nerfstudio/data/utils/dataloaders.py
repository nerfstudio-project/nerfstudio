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

"""
Code for sampling images from a dataset of images.
"""

import random
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from rich.progress import Console, track
from torch.utils.data import Dataset, default_collate
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_dict_to_torch

CONSOLE = Console(width=120)


def _collate_images(train_pixel_sampler, train_ray_generator, dataset_outputs):
    with torch.no_grad():
        batch = default_collate(dataset_outputs)
        batch = train_pixel_sampler.sample(batch)
        ray_indices = batch["indices"]
        ray_bundle = train_ray_generator(ray_indices)
        return ray_bundle.origins,ray_bundle.directions,ray_bundle.pixel_area,ray_bundle.camera_indices, batch


class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.dataparser_outputs.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

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
            fx=self.cameras.fx[image_idx : image_idx + 1],
            fy=self.cameras.fy[image_idx : image_idx + 1],
            cx=self.cameras.cx[image_idx : image_idx + 1],
            cy=self.cameras.cy[image_idx : image_idx + 1],
            height=self.cameras.image_height[image_idx : image_idx + 1],
            width=self.cameras.image_width[image_idx : image_idx + 1],
            camera_to_worlds=self.cameras.camera_to_worlds[image_idx : image_idx + 1],
            distortion_params=distortion_params,
            camera_type=self.cameras.camera_type[image_idx : image_idx + 1],
        )
        return camera

    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx)
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            ray_bundle, batch = self.get_data_from_image_idx(image_idx)
            self.count += 1
            return ray_bundle, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
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
