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

from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.nn import Parameter

from nerfactory.cameras.rays import RayBundle
from nerfactory.dataloaders.eval import FixedIndicesEvalDataloader
from nerfactory.dataloaders.image_dataset import ImageDataset, PanopticImageDataset
from nerfactory.dataloaders.image_sampler import CacheImageSampler
from nerfactory.dataloaders.pixel_sampler import PixelSampler
from nerfactory.dataloaders.structs import DatasetInputs
from nerfactory.models.modules.ray_generator import RayGenerator
from nerfactory.utils import profiler
from nerfactory.utils.config import DataloaderConfig
from nerfactory.utils.misc import IterableWrapper, instantiate_from_dict_config


class Dataloader(nn.Module):
    """Second version of the dataloader class (V2)

    This version of the dataloader is designed to subsume both the train and eval dataloaders,
    especially since this may contain learnable parameters which need to be shared across the train
    and test dataloaders. The idea is that we have setup methods for train and eval separatley and
    this can be a combined train/eval if you want.

    The default should be for the train and eval dataloaders to be the same, but this can be
    overridden. This is needed when there are learned parameters either in your data itself or in
    the way some of the data (that we will pass to the renderer / Field) was generated. In these
    cases, we want these parameters to be accessible by both the train and eval dataloaders, hence
    why you would want them to be in the same dataloader.

    An instance where you may want to have only the eval dataloader is if you are doing evaluation
    and don't have the dataset used to train the model.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: returns an iterator of the train dataloader
        next_train: will be called on __next__() for the training iterator

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: returns an iterator of the eval dataloader
        next_eval: will be called on __next__() for the eval iterator

    Args:
        image_sampler (ImageSampler): image sampler
        pixel_sampler (PixelSampler): pixel sampler
        ray_generator (RayGenerator): ray generator
        use_train (bool): whether this is being used for training
        use_eval (bool): whether this is being used for evaluation

    Attributes:
        image_sampler (ImageSampler): image sampler
        pixel_sampler (PixelSampler): pixel sampler
        train_count (int): number of times train has been called
        eval_count (int): number of times eval has been called
    """

    def __init__(
        self,
        use_train: bool,
        use_eval: bool,
    ):
        super().__init__()
        self.use_train = use_train
        self.use_eval = use_eval
        self.train_count = 0
        self.eval_count = 0
        assert use_train or use_eval
        if use_train:
            self.setup_train()
        if use_eval:
            self.setup_eval()

    def forward(self):
        """Dummy forward method"""
        raise NotImplementedError

    def iter_train(self) -> IterableWrapper:
        """Returns an iterator that executes the self.next_train function"""
        self.train_count = 0
        return IterableWrapper(self, self.next_train)

    def iter_eval(self) -> IterableWrapper:
        """Returns an iterator that executes the self.next_eval function"""
        self.eval_count = 0
        return IterableWrapper(self, self.next_eval)

    @abstractmethod
    def setup_train(self):
        """Sets up the dataloader for training"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the dataloader for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self) -> Tuple:
        """Returns the next batch of data from the train dataloader.

        This will be a tuple of all the information that this dataloader outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self) -> Tuple:
        """Returns the next batch of data from the eval dataloader.

        This will be a tuple of all the information that this dataloader outputs.
        """
        raise NotImplementedError

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the dataloader.

        Returns:
            A list of dictionaries containing the dataloader's param groups.
        """
        return {}


class VanillaDataloader(Dataloader):  # pylint: disable=abstract-method
    """Basic stored dataloader implementation for instant-ngp test run"""

    def __init__(
        self,
        image_dataset_type: str,
        train_datasetinputs: DatasetInputs,
        train_num_rays_per_batch: int,
        train_num_images_to_sample_from: int,
        eval_datasetinputs: DatasetInputs,
        eval_image_indices: Union[List[int], ListConfig],  # TODO(ethan): get rid of this hydra ListConfig nonsense
        eval_num_rays_per_chunk: int,
        device: Union[torch.device, str] = "cpu",
        **kwargs  # pylint: disable=unused-argument
    ):
        self.image_dataset_type = image_dataset_type
        self.train_datasetinputs = train_datasetinputs
        self.train_num_rays_per_batch = train_num_rays_per_batch
        self.train_num_images_to_sample_from = train_num_images_to_sample_from
        self.eval_datasetinputs = eval_datasetinputs
        self.eval_image_indices = eval_image_indices
        self.eval_num_rays_per_chunk = eval_num_rays_per_chunk
        self.device = device
        use_train = self.train_datasetinputs is not None
        use_eval = self.eval_datasetinputs is not None
        super().__init__(use_train, use_eval)

    def setup_train(self):
        """Sets up the dataloader for training"""
        if self.image_dataset_type == "rgb":
            self.train_image_dataset = ImageDataset(**self.train_datasetinputs.as_dict())
        elif self.image_dataset_type == "panoptic":
            self.train_image_dataset = PanopticImageDataset(**self.train_datasetinputs.as_dict())
        self.train_image_sampler = CacheImageSampler(
            self.train_image_dataset, num_images_to_sample_from=self.train_num_images_to_sample_from, device=self.device
        )  # TODO(ethan): pass this in
        self.iter_train_image_sampler = iter(self.train_image_sampler)
        self.train_pixel_sampler = PixelSampler(self.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(
            self.train_datasetinputs.intrinsics, self.train_datasetinputs.camera_to_world
        )

    def setup_eval(self):
        """Sets up the dataloader for evaluation"""
        if self.image_dataset_type == "rgb":
            self.eval_image_dataset = ImageDataset(**self.eval_datasetinputs.as_dict())
        elif self.image_dataset_type == "panoptic":
            self.eval_image_dataset = PanopticImageDataset(**self.eval_datasetinputs.as_dict())
        self.eval_dataloader = FixedIndicesEvalDataloader(
            image_dataset=self.eval_image_dataset,
            intrinsics=self.eval_datasetinputs.intrinsics,
            camera_to_world=self.eval_datasetinputs.camera_to_world,
            num_rays_per_chunk=self.eval_num_rays_per_chunk,
            image_indices=self.eval_image_indices,
            device=self.device,
        )

    def next_train(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader.
        The RayBundle can be shaped in whatever way."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_sampler)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator.forward(ray_indices)
        return ray_bundle, batch

    def next_eval(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader.
        The RayBundle should be shaped like an image."""
        self.eval_count += 1
        camera_ray_bundle, batch = next(self.eval_dataloader)
        return camera_ray_bundle, batch


@profiler.time_function
def setup_dataloader(config: DataloaderConfig, device: str, test_mode=False) -> Dataloader:
    """Setup the dataloader."""

    dataset_train = instantiate_from_dict_config(config.train_dataset)
    train_datasetinputs = dataset_train.get_dataset_inputs(split="train")
    dataset_eval = instantiate_from_dict_config(config.eval_dataset)
    eval_datasetinputs = dataset_eval.get_dataset_inputs(split="val" if not test_mode else "test")

    dataloader: Dataloader = instantiate_from_dict_config(
        DictConfig(config),
        train_datasetinputs=train_datasetinputs,
        eval_datasetinputs=eval_datasetinputs,
        device=device,
    )
    dataloader.to(device)
    return dataloader
