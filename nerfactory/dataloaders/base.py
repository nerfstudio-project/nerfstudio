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
from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn import Parameter

from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.dataloaders.eval import FixedIndicesEvalDataloader
from nerfactory.dataloaders.image_dataset import ImageDataset, PanopticImageDataset
from nerfactory.dataloaders.image_sampler import CacheImageSampler
from nerfactory.dataloaders.pixel_sampler import PixelSampler
from nerfactory.models.modules.ray_generator import RayGenerator
from nerfactory.utils.misc import IterableWrapper


class Dataloader(nn.Module):
    """Generic dataloader's abstract class

    This version of the dataloader is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test dataloaders. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This dataloader's next_train and next_eval methods will return 2 things:
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

    Args:
        use_train (bool): whether this dataloader is being used for training
        use_eval (bool): whether this dataloader is being used for evaluation

    Attributes:
        use_train (bool): whether or not we are using train
        use_eval (bool): whether or not we are using eval
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually

        Additional attributes specific to each subclass are defined in the setuo_train and setup_eval
        functions.

    """

    def __init__(self, use_train: bool, use_eval: bool):
        """Constructor for the Dataloader class.

        Subclassed Dataloaders will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__(use_train, use_eval) BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
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
        |    for ray_bundle, batch in dataloader.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our Dataloader instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in dataloader.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our Dataloader instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the dataloader for training.

        Here you will define any subclass specific object attributes from the attribute"""
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

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the dataloader.

        Returns:
            A list of dictionaries containing the dataloader's param groups.
        """
        return {}


class VanillaDataloader(Dataloader):  # pylint: disable=abstract-method
    """Basic stored dataloader implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions."""

    config: cfg.DataloaderConfig

    def __init__(
        self,
        config: cfg.DataloaderConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: bool = False,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device

        dataset_train = config.train_dataset.setup()
        self.train_datasetinputs = dataset_train.get_dataset_inputs(split="train")
        self.train_cameras = self.train_datasetinputs.cameras.to(device)
        if config.eval_dataset is not None:
            dataset_eval = config.eval_dataset.setup()
        else:
            logging.info("No eval dataset specified so using train dataset for eval.")
            dataset_eval = dataset_train
        self.config.eval_datasetinputs = dataset_eval.get_dataset_inputs(split="val" if not test_mode else "test")
        self.eval_cameras = self.config.eval_datasetinputs.cameras.to(device)
        use_train = self.train_datasetinputs is not None
        use_eval = self.config.eval_datasetinputs is not None
        super().__init__(use_train, use_eval)

    def setup_train(self):
        """Sets up the dataloader for training"""
        if self.config.image_dataset_type == "rgb":
            self.train_image_dataset = ImageDataset(**self.train_datasetinputs.as_dict())
        elif self.config.image_dataset_type == "panoptic":
            self.train_image_dataset = PanopticImageDataset(**self.train_datasetinputs.as_dict())
        self.train_image_sampler = CacheImageSampler(
            self.train_image_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            device=self.device,
        )  # TODO(ethan): pass this in
        self.iter_train_image_sampler = iter(self.train_image_sampler)
        self.train_pixel_sampler = PixelSampler(self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_cameras)

    def setup_eval(self):
        """Sets up the dataloader for evaluation"""
        if self.config.image_dataset_type == "rgb":
            self.eval_image_dataset = ImageDataset(**self.config.eval_datasetinputs.as_dict())
        elif self.config.image_dataset_type == "panoptic":
            self.eval_image_dataset = PanopticImageDataset(**self.config.eval_datasetinputs.as_dict())
        self.eval_dataloader = FixedIndicesEvalDataloader(
            image_dataset=self.eval_image_dataset,
            cameras=self.eval_cameras,
            num_rays_per_chunk=self.config.eval_num_rays_per_chunk,
            image_indices=self.config.eval_image_indices,
            device=self.device,
        )

    def next_train(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader.
        The RayBundle can be shaped in whatever way."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_sampler)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader.
        The RayBundle should be shaped like an image."""
        self.eval_count += 1
        camera_ray_bundle, batch = next(self.eval_dataloader)
        return camera_ray_bundle, batch
