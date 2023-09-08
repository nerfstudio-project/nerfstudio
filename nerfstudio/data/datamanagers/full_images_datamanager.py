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
Data manager that outputs cameras / images instead of raybundles

Good for things like gaussian splatting which require full cameras instead of the standard ray
paradigm
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (Any, Callable, Dict, ForwardRef, Generic, List, Literal,
                    Optional, Tuple, Type, Union, cast, get_args, get_origin)
import random
import torch
from rich.progress import Console
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal, TypeVar

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import (DataManager,
                                                           DataManagerConfig)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import \
    BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import (PatchPixelSamplerConfig,
                                            PixelSampler, PixelSamplerConfig)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import (CacheDataloader,
                                               FixedIndicesEvalDataloader,
                                               RandIndicesEvalDataloader)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE

CONSOLE = Console(width=120)

@dataclass
class FullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

class FullImageDatamanager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager far more lightweight since we dont have to do ray generation for things like
    gaussian splatting that dont require it.
    """

    config: FullImageDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

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
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"

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

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
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
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
    
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch
        
        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras)-1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
        data = self.train_dataset.get_data(image_idx)
        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx]
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch
        
        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras)-1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.eval_dataset.get_data(image_idx)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx]
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[int, Cameras, Dict]:
        """Returns the next evaluation batch
        
        Returns a Camera instead of raybundle
        
        TODO (jake-austin): Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras)-1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.eval_dataset.get_data(image_idx)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx]
        return image_idx, camera, data
