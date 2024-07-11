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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import torch
import tyro
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.dataloaders import (  # , RayBatchStream
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for the cached dataloader.
    Args:
        batch: Batch of samples from the dataset.
    Returns:
        Collated batch.
    """
    images = []
    imgdata_lists = defaultdict(list)
    for data in batch:
        image = data.pop("image")
        images.append(image)
        topop = []
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                # if the value has same height and width as the image, assume that it should be collated accordingly.
                if len(val.shape) >= 2 and val.shape[:2] == image.shape[:2]:
                    imgdata_lists[key].append(val)
                    topop.append(key)
        # now that iteration is complete, the image data items can be removed from the batch
        for key in topop:
            del data[key]

    new_batch = nerfstudio_collate(batch)
    new_batch["image"] = images
    new_batch.update(imgdata_lists)

    return new_batch


@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    masks_on_gpu: bool = False
    """Process masks on GPU for speed at the expense of memory, if True."""
    images_on_gpu: bool = False
    """Process images on GPU for speed at the expense of memory, if True."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. 'rays': This will contain the rays or camera we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
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
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset
        includes_time (bool): whether the dataset includes time information

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    includes_time: bool = False

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
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
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
        as __iter__ and __next__ methods respectively.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectively.

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

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        """Returns the next batch of data from the train data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        """Returns the next batch of data from the eval data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray/camera for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Retrieve the next eval image.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the step number, the ray/camera for the image, and a dictionary of
            additional batch information such as the groundtruth image.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        raise NotImplementedError

    @abstractmethod
    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        raise NotImplementedError

    @abstractmethod
    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(DataManagerConfig):
    """A basic data manager for a ray-based model"""

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=BlenderDataParserConfig)
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    patch_size: int = 1
    """Size of patch to sample from. If > 1, patch-based sampling will be used."""
    prefetch_factor: int = 1
    """The limit number of batches a worker will start loading once an iterator is created. 
    More details are described here: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""
    dataloader_num_workers: int = 2
    """The number of workers performing the dataloading from either disk/RAM, which 
    includes undistortion, pixel sampling, ray generation, collating, etc."""
    use_ray_train_dataloader: bool = True
    """Allows parallelization of the dataloading process with multiple workers."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for this field.
    camera_optimizer: tyro.conf.Suppress[Optional[CameraOptimizerConfig]] = field(default=None)
    """Deprecated, has been moved to the model config."""
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)
    """Specifies the pixel sampler used to sample pixels from images."""

    def __post_init__(self):
        """Warn user of camera optimizer change."""
        if self.camera_optimizer is not None:
            import warnings

            CONSOLE.print(
                "\nCameraOptimizerConfig has been moved from the DataManager to the Model.\n", style="bold yellow"
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)

import concurrent.futures
import math
import multiprocessing
import random
from typing import Sized

from torch.utils.data import Dataset

from nerfstudio.utils.misc import get_dict_to_torch


class RayBatchStream(torch.utils.data.IterableDataset):
    def __init__(
        self,
        input_dataset: Dataset,
        datamanager_config: DataManagerConfig,
        num_images_to_sample_from: int = -1,  # passed in from VanillaDataManager
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        num_image_load_threads: int = 4,
        cache_all_n_shard_per_worker: bool = True,  # When False, always getting Killed/bugs for some reason... why?
        # when cache_all_n_shard_per_worker True, getting killed because caching everything is not good
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.input_dataset = input_dataset
        assert isinstance(self.input_dataset, Sized)

        # self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))
        # self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.num_images_to_sample_from = num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_image_load_threads = num_image_load_threads  # kwargs.get("num_workers", 4) # nb only 4 in defaults
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device

        self.datamanager_config = datamanager_config
        self.pixel_sampler: PixelSampler = None
        self.ray_generator: RayGenerator = None
        self._cached_collated_batch = None
        """_cached_collated_batch contains a collated batch of images for a specific worker that's ready for pixel sampling."""
        self.cache_all_n_shard_per_worker = cache_all_n_shard_per_worker
        """If True, _cached_collated_batch is populated with a subset of the dataset assigned to each worker during the iteration process."""

    def _get_pixel_sampler(self, dataset: "TDataset", num_rays_per_batch: int) -> PixelSampler:
        """copy-pasta from VanillaDataManager."""
        from nerfstudio.cameras.cameras import CameraType
        from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSamplerConfig

        if self.datamanager_config.patch_size > 1 and type(self.datamanager_config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.datamanager_config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.datamanager_config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def _get_batch_list(self, indices=None):
        """Returns a list representing a single batch from the dataset attribute.
        Each item of the list is a dictionary with dict_keys(['image_idx', 'image']) representing 1 image.
        This function is used to sample and load images from disk/RAM and is only called in _get_collated_batch
        The length of the list is equal to the (# of training images) / (num_workers)"""

        assert isinstance(self.input_dataset, Sized)
        if indices is None:
            # Note: self.num_images_to_sample_from is usually -1, but _get_batch_list is usually called with indices != None.
            # _get_batch_list is used by _get_collated_batch, whose indices = some partition of the dataset
            indices = random.sample(range(len(self.input_dataset)), k=self.num_images_to_sample_from)
        batch_list = []
        results = []

        # num_threads = int(self.num_ds_load_threads) * 4
        num_threads = (
            int(self.num_image_load_threads)
            if not self.cache_all_n_shard_per_worker
            else 4 * int(self.num_image_load_threads)
        )
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)
        # print('num_threads', num_threads)

        # NB: this is I/O heavy because we are going to disk and reading an image filename
        # hence multi-threaded inside the worker
        from tqdm.auto import tqdm

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.input_dataset.__getitem__, idx)
                results.append(res)

            # for res in track(results, description="Loading data batch", transient=True):
            # for res in tqdm(results, desc='_get_batch_list'):
            if self.cache_all_n_shard_per_worker:
                results = tqdm(results)
            for res in results:
                batch_list.append(res.result())
        return batch_list

    def _get_collated_batch(self, indices=None):
        """Takes the output of _get_batch_list and collates them with nerfstudio_collate()
        Note: dict is an instance of collections.abc.Mapping

        The resulting output is collated_batch: a dictionary with dict_keys(['image_idx', 'image'])
        collated_batch['image_idx'] is tensor with shape torch.Size([per_worker])
        collated_batch['image'] is tensor with shape torch.Size([per_worker, height, width, 3])
        """
        batch_list = self._get_batch_list(indices=indices)
        # print(type(batch_list[0])) # prints <class 'dict'>
        # print(self.collate_fn) # prints nerfstudio_collate
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        """This implementation has every worker cache the indices of the images they will use to generate rays."""
        dataset_indices = list(
            range(len(self.input_dataset))
        )  # this_indices has length = numTrainingImages, at first it is the whole training dataset, but it gets partitioned into equal chunks
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # if we have multiple processes
            per_worker = int(math.ceil(len(dataset_indices) / float(worker_info.num_workers)))
            slice_start = worker_info.id * per_worker
        else:  # we only have a single process
            per_worker = len(self.input_dataset)
            slice_start = 0
        worker_indices = dataset_indices[
            slice_start : slice_start + per_worker
        ]  # the indices of the datapoints in the dataset this worker will load
        r = random.Random(3301)
        num_rays_per_loop = self.datamanager_config.train_num_rays_per_batch # default train_num_rays_per_batch is 4096
        worker_pixel_sampler = self._get_pixel_sampler(self.input_dataset, num_rays_per_loop)
        if self.ray_generator is None:
            self.ray_generator = RayGenerator(self.input_dataset.cameras)#.to(self.device))
        i = 0
        while True:
            if i % 5 == 0:
                r.shuffle(worker_indices)
                image_indices = worker_indices[:self.num_images_to_sample_from] # get a total of 'num_images_to_sample_from' image indices 
                
                # self._get_collated_batch is slow because it is going to disk to retreive an image many times to create a batch of images.
                collated_batch = self._get_collated_batch(image_indices)
            i += 1
            """
            Here, the variable 'batch' refers to the output of our pixel sampler.
                - batch is a dict_keys(['image', 'indices'])
                - batch['image'] returns a pytorch tensor with shape `torch.Size([4096, 3])` , where 4096 = num_rays_per_batch. Note: each row in this tensor represents the RGB values as floats in [0, 1] of the pixel the ray goes through. The info of what specific image index that pixel belongs to is stored within batch[’indices’]
                - batch['indices'] returns a pytorch tensor `torch.Size([4096, 3])` tensor where each row represents (image_index=camera_index, pixelRow, pixelCol)
            What the pixel_sampler does (for variable_res_collate) is that it loops though each image, samples pixel within the mask, 
            and returns them as the variable `indices` which has shape torch.Size([4096, 3]), where each row represents a pixel (image_idx, pixelRow, pixelCol)
            """
            batch = worker_pixel_sampler.sample(collated_batch) # the pixel_sampler will sample num_rays_per_batch pixels.
            ray_indices = batch["indices"]
            ray_bundle = self.ray_generator(ray_indices)
            yield ray_bundle, batch


def identity(x):
    return x

class VanillaDataManager(DataManager, Generic[TDataset]):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: VanillaDataManagerConfig,
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
        if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
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
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")

        if self.config.use_ray_train_dataloader:
            import torch.multiprocessing as mp
            mp.set_start_method("spawn")
            
            self.raybatch_stream = RayBatchStream(
                input_dataset=self.train_dataset,
                datamanager_config=self.config,
                num_images_to_sample_from=100, # self.config.train_num_images_to_sample_from,
                device=self.device,
                collate_fn=self.config.collate_fn,
            )
            self.ray_dataloader = torch.utils.data.DataLoader(
                self.raybatch_stream,
                batch_size=1,
                num_workers=self.config.dataloader_num_workers,
                prefetch_factor=self.config.prefetch_factor,
                shuffle=False,
                pin_memory=False,
                # Our dataset does batching / collation
                collate_fn=identity,
                # pin_memory_device=self.device, # did not actually speed up my implementation
            )
            self.iter_train_image_dataloader = None
            self.iter_train_raybundles = iter(self.ray_dataloader)
        else:
            self.iter_train_raybundles = None
            self.train_image_dataloader = CacheDataloader(
                self.train_dataset,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4
                if self.config.dataloader_num_workers == -1
                else self.config.dataloader_num_workers,
                prefetch_factor=2
                if self.config.dataloader_prefetch_size == -1
                else self.config.dataloader_prefetch_size,
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
        self.train_count += 1
        if self.config.use_ray_train_dataloader:
            ret = next(self.iter_train_raybundles)
            assert len(ret) == 1, f"batch size should be one {len(ret)}"
            ray_bundle, batch = ret[0]
            # ray_bundle = RayBundle.from_dict(ray_bundle_dict)
            ray_bundle = ray_bundle.to(self.device)
        else:
            image_batch = next(self.iter_train_image_dataloader)
            assert self.train_pixel_sampler is not None
            assert isinstance(image_batch, dict)
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

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
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
