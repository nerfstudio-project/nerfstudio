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
Generative datamanager.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from typing_extensions import Literal

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.pigan_dataparser import PiganDataParserConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, DataManagerConfig, DataManager

from nerfstudio.cameras.rays import RayBundle

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import (
    GanDataloader,
    RandIndicesEvalDataloader,
)
import tyro

CONSOLE = Console(width=120)

@dataclass
class GenerativeDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: GenerativeDataManager)
    """Target class to instantiate."""
    dataparser = PiganDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    # train_num_rays_per_batch: int = 1024
    # """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = 4
    """Number of images to sample during training iteration."""
    # train_num_times_to_repeat_images: int = -1
    # """When not training on all images, number of iterations before picking new
    # images. If -1, never pick new images."""
    # eval_num_rays_per_batch: int = 1024
    # """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = 1
    """Number of images to sample during eval iteration."""
    # eval_num_times_to_repeat_images: int = -1
    # """When not evaluating on all images, number of iterations before picking
    # new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    # camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    # """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    # Record3D."""
    # collate_fn = staticmethod(nerfstudio_collate)
    # """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    # patch_size: int = 1
    # """Size of patch to sample from. If >1, patch-based sampling will be used."""


class GenerativeDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: GenerativeDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset
    train_dataparser_outputs: DataparserOutputs
    # train_pixel_sampler: Optional[PixelSampler] = None
    # eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        # self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return InputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    # def _get_pixel_sampler(  # pylint: disable=no-self-use
    #     self, dataset: InputDataset, *args: Any, **kwargs: Any
    # ) -> PixelSampler:
    #     """Infer pixel sampler to use."""
    #     if self.config.patch_size > 1:
    #         return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

    #     # If all images are equirectangular, use equirectangular pixel sampler
    #     is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
    #     if is_equirectangular.all():
    #         return EquirectangularPixelSampler(*args, **kwargs)
    #     # Otherwise, use the default pixel sampler
    #     if is_equirectangular.any():
    #         CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
    #     return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = GanDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
        )
        # self.train_image_dataloader = CacheDataloader(
        #     self.train_dataset,
        #     num_images_to_sample_from=self.config.train_num_images_to_sample_from,
        #     num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
        #     device=self.device,
        #     num_workers=self.world_size * 4,
        #     pin_memory=True,
        #     collate_fn=self.config.collate_fn,
        # )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        # self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        # self.train_camera_optimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.train_dataset.cameras.size, device=self.device
        # )
        # self.train_ray_generator = RayGenerator(
        #     self.train_dataset.cameras.to(self.device),
        #     self.train_camera_optimizer,
        # )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = GanDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
        )
        # self.eval_image_dataloader = CacheDataloader(
        #     self.eval_dataset,
        #     num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
        #     num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
        #     device=self.device,
        #     num_workers=self.world_size * 4,
        #     pin_memory=True,
        #     collate_fn=self.config.collate_fn,
        # )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        # self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        # self.eval_camera_optimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.eval_dataset.cameras.size, device=self.device
        # )
        # self.eval_ray_generator = RayGenerator(
        #     self.eval_dataset.cameras.to(self.device),
        #     self.eval_camera_optimizer,
        # )
        # # for loading full images
        # self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
        #     input_dataset=self.eval_dataset,
        #     device=self.device,
        #     num_workers=self.world_size * 4,
        # )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        for camera_ray_bundle, batch in self.iter_train_image_dataloader:
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        # ray_bundle, batch = next(self.iter_train_image_dataloader)
        # # FIXME - fixed를 쓰면 애초에 ray_bundle과 batch를 같이 return해주게 할 수 있다. 이것에 따라서 
        # # datamanger도 수정되어야 한다.
        # return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        ray_bundle, batch = next(self.iter_eval_image_dataloader)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.dataparser.num_rays_per_image * self.config.train_num_images_to_sample_from

    def get_eval_rays_per_batch(self) -> int:
        return self.dataparser.num_rays_per_image * self.config.eval_num_images_to_sample_from

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        # camera_opt_params = list(self.train_camera_optimizer.parameters())
        # if self.config.camera_optimizer.mode != "off":
        #     assert len(camera_opt_params) > 0
        #     param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        # else:
        #     assert len(camera_opt_params) == 0

        return param_groups
