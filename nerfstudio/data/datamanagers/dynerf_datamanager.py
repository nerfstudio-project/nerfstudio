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
Datamanager for Neural 3D Video Synthesis Dataset.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Sized, Tuple, Type, Union

import h5py
import torch
from rich.progress import track
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    TDataset,
    VanillaDataManagerConfig,
    variable_res_collate,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.dynerf_dataparser import DyNeRFDataParserConfig
from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE


class VideoDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        num_frames: int,
        num_cameras: int,
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        assert isinstance(self.dataset, Sized)
        assert len(self.dataset) == self.num_frames * self.num_cameras

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        frame_idx = random.randint(0, self.num_frames - 1)
        # select num_cameras images at timestamp t
        indices = list(range(frame_idx * self.num_cameras, (frame_idx + 1) * self.num_cameras))
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list, frame_idx

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list, frame_idx = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        collated_batch["frame_idx"] = frame_idx
        return collated_batch

    def __iter__(self):
        while True:
            collated_batch = self._get_collated_batch()
            yield collated_batch


@dataclass
class DyNeRFDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: DyNeRFDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = DyNeRFDataParserConfig()
    """Specifies the dataparser used to unpack the data."""

    importance_on_gpu: bool = False
    """Process importance weights on GPU for speed at the expense of memory, if True."""
    precompute_device: Union[torch.device, str] = "cpu"
    """Precompute ISG / IST weights on the given device."""
    isg_gamma: float = 2e-2
    """Hyperparameter of ISG"""
    ist_alpha: float = 0.1
    """Hyperparameter of IST"""
    ist_shift: int = 25
    """Hyperparameter of IST"""
    isg_step: int = 0
    """Steps before starting to use ISG weights. -1 to disable."""
    ist_step: int = 250000
    """Steps before starting to use IST weights. -1 to disable."""


class DyNeRFDataManager(DataManager, Generic[TDataset]):
    """DataManager for Neural 3D Video Synthesis Dataset"""

    config: DyNeRFDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: DyNeRFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.dataset_type: Type[TDataset] = kwargs.get("_dataset_type", getattr(TDataset, "__default__"))
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1
        self.includes_time = self.dataparser.includes_time
        if test_mode != "inference":  # Avoid opening images in inference
            self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
            self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(
                split=self.test_split
            )
            self.train_num_frames, self.train_num_cameras = self.train_dataparser_outputs.metadata[
                "shape_before_flatten"
            ]
            camera_heights = self.train_dataparser_outputs.cameras.height[: self.train_num_cameras, 0]
            camera_widths = self.train_dataparser_outputs.cameras.width[: self.train_num_cameras, 0]
            if len(camera_heights) > 1:
                for height, width in zip(camera_heights[1:], camera_widths[1:]):
                    if camera_heights[0] != height or camera_widths[0] != width:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
            self.train_dataset = self.create_train_dataset()
            self.eval_dataset = self.create_eval_dataset()
            self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
            if self.config.masks_on_gpu is True:
                self.exclude_batch_keys_from_device.remove("mask")
            if self.config.images_on_gpu is True:
                self.exclude_batch_keys_from_device.remove("image")
            if self.config.importance_on_gpu is True:
                self.exclude_batch_keys_from_device.remove("importance")

            self.isg_cache = self.config.data / f"x{self.dataparser.downscale_factor}" / "isg_cache.h5"
            self.ist_cache = self.config.data / f"x{self.dataparser.downscale_factor}" / "ist_cache.h5"
            self.importance_mode: Literal["none", "isg", "ist"] = "none"
            if self.config.isg_step >= 0:
                self.dataparser.precompute_isg(
                    self.isg_cache, self.train_dataparser_outputs, self.config.isg_gamma, self.config.precompute_device
                )
                self.isg_cache_f = h5py.File(self.isg_cache)
            if self.config.ist_step >= 0:
                self.dataparser.precompute_ist(
                    self.ist_cache,
                    self.train_dataparser_outputs,
                    self.config.ist_alpha,
                    self.config.ist_shift,
                    self.config.precompute_device,
                )
                self.ist_cache_f = h5py.File(self.ist_cache)
        super().__init__()

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            raise NotImplementedError
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            raise NotImplementedError
        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular, num_rays_per_batch=num_rays_per_batch
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = VideoDataloader(
            self.train_dataset,
            self.train_num_frames,
            self.train_num_cameras,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        if self.train_count == self.config.isg_step:
            self.importance_mode = "isg"
        if self.train_count == self.config.ist_step:
            self.importance_mode = "ist"
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        frame_idx = image_batch["frame_idx"]
        # TODO: support importance weights for camera_res_scale_factor < 1.0
        if self.importance_mode == "isg":
            importance_weights = torch.empty(
                self.train_num_cameras, image_batch["image"].shape[1], image_batch["image"].shape[2]
            )
            for camera in range(self.train_num_cameras):
                importance_weights[camera] = torch.from_numpy(self.isg_cache_f[f"weights/{camera}"][frame_idx])
            importance_weights = torch.tensor(importance_weights)
            importance_weights /= importance_weights.sum()
            image_batch["importance"] = importance_weights
        elif self.importance_mode == "isg":
            importance_weights = torch.empty(
                self.train_num_cameras, image_batch["image"].shape[1], image_batch["image"].shape[2]
            )
            for camera in range(self.train_num_cameras):
                importance_weights[camera] = torch.from_numpy(self.ist_cache_f[f"weights/{camera}"][frame_idx])
            importance_weights = torch.tensor(importance_weights)
            importance_weights /= importance_weights.sum()
            image_batch["importance"] = importance_weights
        del image_batch["frame_idx"]
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
