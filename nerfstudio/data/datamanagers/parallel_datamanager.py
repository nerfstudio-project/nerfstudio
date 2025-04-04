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
Parallel data manager that generates training data in multiple python processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, TDataset, VanillaDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.data.utils.dataloaders import (
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
    RayBatchStream,
    variable_res_collate,
)
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ParallelDataManagerConfig(VanillaDataManagerConfig):
    """Config for a `ParallelDataManager` which reads data in multiple processes"""

    _target: Type = field(default_factory=lambda: ParallelDataManager)
    """Target class to instantiate."""
    load_from_disk: bool = False
    """If True, conserves RAM memory by loading images from disk.
    If False, caches all the images as tensors to RAM and loads from RAM."""
    dataloader_num_workers: int = 4
    """The number of workers performing the dataloading from either disk/RAM, which 
    includes collating, pixel sampling, unprojecting, ray generation etc."""
    prefetch_factor: int = 10
    """The limit number of batches a worker will start loading once an iterator is created. 
    More details are described here: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""
    cache_compressed_images: bool = False
    """If True, cache raw image files as byte strings to RAM."""


class ParallelDataManager(DataManager, Generic[TDataset]):
    """Data manager implementation for parallel dataloading.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ParallelDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: ParallelDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            assert torch.multiprocessing.get_start_method() == "spawn", 'start method must be "spawn"'
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")
        # Setup our collate function (same as base_datamanager.py)
        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height or True:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[ParallelDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is ParallelDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is ParallelDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is ParallelDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training."""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation."""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    def setup_train(self):
        self.train_raybatchstream = RayBatchStream(
            input_dataset=self.train_dataset,
            num_rays_per_batch=self.config.train_num_rays_per_batch,
            num_images_to_sample_from=(
                50
                if self.config.load_from_disk and self.config.train_num_images_to_sample_from == float("inf")
                else self.config.train_num_images_to_sample_from
            ),
            num_times_to_repeat_images=(
                10
                if self.config.load_from_disk and self.config.train_num_times_to_repeat_images == float("inf")
                else self.config.train_num_times_to_repeat_images
            ),
            device=self.device,
            collate_fn=variable_res_collate,
            load_from_disk=self.config.load_from_disk,
            custom_ray_processor=self.custom_ray_processor,
        )
        self.train_ray_dataloader = DataLoader(
            self.train_raybatchstream,
            batch_size=1,
            num_workers=self.config.dataloader_num_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=False,
            collate_fn=identity_collate,  # Our dataset handles batching / collation of rays
        )
        self.iter_train_raybundles = iter(self.train_ray_dataloader)

    def setup_eval(self):
        self.eval_raybatchstream = RayBatchStream(
            input_dataset=self.eval_dataset,
            num_rays_per_batch=self.config.train_num_rays_per_batch,
            num_images_to_sample_from=(
                50
                if self.config.load_from_disk and self.config.eval_num_images_to_sample_from == float("inf")
                else self.config.eval_num_images_to_sample_from
            ),
            num_times_to_repeat_images=(
                10
                if self.config.load_from_disk and self.config.eval_num_times_to_repeat_images == float("inf")
                else self.config.eval_num_times_to_repeat_images
            ),
            device=self.device,
            collate_fn=variable_res_collate,
            load_from_disk=True,
            custom_ray_processor=self.custom_ray_processor,
        )
        self.eval_ray_dataloader = DataLoader(
            self.eval_raybatchstream,
            batch_size=1,
            num_workers=0,  # This must be 0 otherwise there is a crash when trying to pickle custom_ray_processor
            shuffle=False,
            collate_fn=identity_collate,  # Our dataset handles batching / collation of rays
        )
        self.iter_eval_raybundles = iter(self.eval_ray_dataloader)
        self.image_eval_dataloader = RandIndicesEvalDataloader(  # this is used for ns-eval
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(  # this is used by ns-render
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        ray_bundle, batch = next(self.iter_train_raybundles)[0]
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        ray_bundle, batch = next(self.iter_train_raybundles)[0]
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Retrieve the next eval image."""
        for camera, batch in self.image_eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def custom_ray_processor(self, ray_bundle: RayBundle, batch: Dict) -> Tuple[RayBundle, Dict]:
        """An API to add latents, metadata, or other further customization to the RayBundle dataloading process that is parallelized"""
        return ray_bundle, batch
