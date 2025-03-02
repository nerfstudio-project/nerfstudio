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
Parallel data manager that outputs cameras / images instead of raybundles.
"""

from __future__ import annotations

import random
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Tuple, Type, Union, cast, get_args, get_origin

import fpsample
import numpy as np
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import DataManager, TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.data.utils.dataloaders import ImageBatchStream, undistort_view
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


class ParallelFullImageDatamanager(DataManager, Generic[TDataset]):
    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
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
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time

        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = self.sample_train_cameras()
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

        super().__init__()

    def sample_train_cameras(self):
        """Return a list of camera indices sampled using the strategy specified by
        self.config.train_cameras_sampling_strategy"""
        num_train_cameras = len(self.train_dataset)
        if self.config.train_cameras_sampling_strategy == "random":
            if not hasattr(self, "random_generator"):
                self.random_generator = random.Random(self.config.train_cameras_sampling_seed)
            indices = list(range(num_train_cameras))
            self.random_generator.shuffle(indices)
            return indices
        elif self.config.train_cameras_sampling_strategy == "fps":
            if not hasattr(self, "train_unsampled_epoch_count"):
                np.random.seed(self.config.train_cameras_sampling_seed)  # fix random seed of fpsample
                self.train_unsampled_epoch_count = np.zeros(num_train_cameras)
            camera_origins = self.train_dataset.cameras.camera_to_worlds[..., 3].numpy()
            # We concatenate camera origins with weighted train_unsampled_epoch_count because we want to
            # increase the chance to sample camera that hasn't been sampled in consecutive epochs previously.
            # We assume the camera origins are also rescaled, so the weight 0.1 is relative to the scale of scene
            data = np.concatenate(
                (camera_origins, 0.1 * np.expand_dims(self.train_unsampled_epoch_count, axis=-1)), axis=-1
            )
            n = self.config.fps_reset_every
            if num_train_cameras < n:
                CONSOLE.log(
                    f"num_train_cameras={num_train_cameras} is smaller than fps_reset_ever={n}, the behavior of "
                    "camera sampler will be very similar to sampling random without replacement (default setting)."
                )
                n = num_train_cameras
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(data, n, h=3)

            self.train_unsampled_epoch_count += 1
            self.train_unsampled_epoch_count[kdline_fps_samples_idx] = 0
            return kdline_fps_samples_idx.tolist()
        else:
            raise ValueError(f"Unknown train camera sampling strategy: {self.config.train_cameras_sampling_strategy}")

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    def create_eval_dataset(self) -> TDataset:
        """Sets up the data loaders for evaluation"""
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[ParallelFullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is ParallelFullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is ParallelFullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is ParallelFullImageDatamanager:
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

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        self.train_imagebatch_stream = ImageBatchStream(
            input_dataset=self.train_dataset,
            cache_images_type=self.config.cache_images_type,
            sampling_seed=self.config.train_cameras_sampling_seed,
            device=self.device,
            custom_view_processor=self.custom_view_processor,
        )
        self.train_image_dataloader = DataLoader(
            self.train_imagebatch_stream,
            batch_size=1,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=identity_collate,
            # pin_memory_device=self.device, # for some reason if we pin memory, exporting to PLY file doesn't work
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def setup_eval(self):
        self.eval_imagebatch_stream = ImageBatchStream(
            input_dataset=self.eval_dataset,
            cache_images_type=self.config.cache_images_type,
            sampling_seed=self.config.train_cameras_sampling_seed,
            device=self.device,
            custom_view_processor=self.custom_view_processor,
        )
        self.eval_image_dataloader = DataLoader(
            self.eval_imagebatch_stream,
            batch_size=1,
            num_workers=0,
            collate_fn=identity_collate,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        return self.iter_eval_image_dataloader

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

    def get_train_rays_per_batch(self):
        # TODO: fix this to be the resolution of the last image rendered
        return 800 * 800

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        self.train_count += 1
        camera, data = next(self.iter_train_image_dataloader)[0]
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        self.eval_count += 1
        camera, data = next(self.iter_train_image_dataloader)[0]
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        return undistort_view(image_idx, self.eval_dataset, self.config.cache_images_type)

    def custom_view_processor(self, camera: Cameras, data: Dict) -> Tuple[Cameras, Dict]:
        """An API to add latents, metadata, or other further customization an camera-and-image view dataloading process that is parallelized"""
        return camera, data
