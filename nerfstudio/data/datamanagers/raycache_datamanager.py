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
Data manager for ray level caching
"""

from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import tyro
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal

import nerfstudio.utils.profiler as profiler
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper

CONSOLE = Console(width=120)


@dataclass
class RayCacheDataManagerConfig(VanillaDataManagerConfig):
    """A datamanager for optimized caching of ray level data."""

    _target: Type = field(default_factory=lambda: RayCacheDataManager)

    # train_num_images_to_sample_from: int = 40
    # """Number of images to sample during training iteration."""
    # train_num_times_to_repeat_images: int = 100
    # """When not training on all images, number of iterations before picking new
    # images. If -1, never pick new images."""
    # eval_num_images_to_sample_from: int = 40
    # """Number of images to sample during eval iteration."""
    # eval_num_times_to_repeat_images: int = 100
    # """When not training on all images, number of iterations before picking new"""

    # ---------- RayCacheDataManager Specific Args ----------#
    num_rays_cached: int = 100000
    """Number of rays to cache / refresh buffer with."""
    train_num_times_to_repeat_ray_cache: int = 50
    """Number of iterations before refreshing the ray cache. 
    Also ends up being number of steps in between camera updates."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(optimizer=AdamOptimizerConfig(weight_decay=0))
    """Camera optimizer config."""


class RayCacheDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """
    Wrapper around the vanilla data manager that will cache raw rays

    Nothing is fundamentally flawed or incorrect about the original datamanager... it just could be quicker.
    """

    config: RayCacheDataManagerConfig

    _ray_cache = None
    _batch_cache = None

    def _refresh_cache(self):
        """Refresh the ray cache and return the next batch."""

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = PixelSampler(self.config.num_rays_cached)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def next_train(self, step: int):
        """Get the next batch of training data."""
        if self._ray_cache is None or step % self.config.train_num_times_to_repeat_ray_cache == 0:
            self._ray_cache, self._batch_cache = super().next_train(step)

        start = random.randint(0, self.config.num_rays_cached - self.config.train_num_rays_per_batch)
        rays = self._ray_cache[start : start + self.config.train_num_rays_per_batch]
        batch = {k: v[start : start + self.config.train_num_rays_per_batch] for k, v in self._batch_cache.items()}

        if step % self.config.train_num_times_to_repeat_ray_cache == 0:
            self._ray_cache = self._ray_cache.detach()
            self._batch_cache = {k: v.detach() for k, v in self._batch_cache.items()}

        return rays, batch
