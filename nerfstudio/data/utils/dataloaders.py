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
Code for sampling images from a dataset of images.
"""

import random
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, default_collate
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.datasets import InputDataset
from nerfstudio.utils.misc import get_dict_to_torch


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images.
        device: Device to perform computation.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: int = -1,
        num_times_to_repeat_images: int = 0,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.dataset = dataset
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = num_images_to_sample_from == -1
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            self.cached_collated_batch = self._get_collated_batch()
        super().__init__(dataset=dataset, **kwargs)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_sample_from)

        batch_list = [self.dataset.__getitem__(idx) for idx in indices]
        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = default_collate(batch_list)
        collated_batch = get_dict_to_torch(collated_batch, device=self.device, exclude=["image"])
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


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
        self.cameras = input_dataset.dataset_inputs.cameras.to(device)
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
            fx=self.cameras.fx if isinstance(self.cameras.fx, float) else self.cameras.fx[image_idx : image_idx + 1],
            fy=self.cameras.fy if isinstance(self.cameras.fy, float) else self.cameras.fy[image_idx : image_idx + 1],
            cx=self.cameras.cx,
            cy=self.cameras.cy,
            camera_to_worlds=self.cameras.camera_to_worlds[image_idx : image_idx + 1],
            distortion_params=distortion_params,
            camera_type=self.cameras.camera_type,
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
