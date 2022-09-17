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
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler

from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.datamanagers.dataloaders import (
    CacheImageDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfactory.datamanagers.datasets import InputDataset
from nerfactory.datamanagers.pixel_sampler import PixelSampler
from nerfactory.models.modules.ray_generator import RayGenerator
from nerfactory.utils.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfactory.utils.misc import IterableWrapper


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:
        1. A Raybundle: This will contain the rays we are sampling, with latents and
            conditionals attached (everything needed at inference)
        2. A "batch" of auxilury information: This will contain the mask, the ground truth
            pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_input_dataset (InputDataset): the input dataset for the train dataset
        eval_input_dataset (InputDataset): the input dataset for the eval dataset

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_input_dataset: Optional[InputDataset] = None
    eval_input_dataset: Optional[InputDataset] = None

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_input_dataset and self.train_input_dataset.dataset_inputs:
            self.setup_train()
        if self.eval_input_dataset and self.eval_input_dataset.dataset_inputs:
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        """Returns the next eval image."""
        raise NotImplementedError

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


class VanillaDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: cfg.VanillaDataManagerConfig

    def __init__(
        self,
        config: cfg.VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: bool = False,
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None

        if config.eval_dataparser is None:
            logging.info("No eval dataset specified so using train dataset for eval.")
            config.eval_dataparser = config.train_dataparser
        self.train_input_dataset = InputDataset(config.train_dataparser, split="train")
        self.eval_input_dataset = InputDataset(config.eval_dataparser, split="val" if not test_mode else "test")
        super().__init__()

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_input_dataset is not None
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.train_input_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True, seed=42
            )
            self.train_image_dataloader = CacheImageDataloader(
                self.train_input_dataset,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                sampler=sampler,
            )  # TODO(ethan): pass this in
        else:
            self.train_image_dataloader = CacheImageDataloader(
                self.train_input_dataset,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
            )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = PixelSampler(self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_input_dataset.dataset_inputs.cameras.to(self.device))

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_input_dataset is not None
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.eval_input_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True, seed=42
            )
            self.eval_image_dataloader = CacheImageDataloader(
                self.eval_input_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                sampler=sampler,
            )  # TODO(ethan): pass this in
        else:
            self.eval_image_dataloader = CacheImageDataloader(
                self.eval_input_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
            )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = PixelSampler(self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_input_dataset.dataset_inputs.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_input_dataset,
            device=self.device,
            num_workers=0 if self.world_size == 1 else self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_input_dataset,
            image_indices=self.config.eval_image_indices,
            device=self.device,
            num_workers=0 if self.world_size == 1 else self.world_size * 4,
        )

        # TODO: eval dataloader should be separate from train
        self.iter_eval_dataloader = iter(self.eval_image_dataloader)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            image_idx = int(camera_ray_bundle.camera_indices[0, 0])
            return image_idx, camera_ray_bundle, batch
