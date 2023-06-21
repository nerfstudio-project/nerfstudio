from __future__ import annotations

import concurrent.futures
import functools
import torch.multiprocessing as mp
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, List, Literal, Optional,
                    Tuple, Type, Union, cast)

import torch
from rich.progress import track
from torch import nn
from torch.nn import Parameter
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager, DataManagerConfig, TDataset, variable_res_collate)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import \
    BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import (EquirectangularPixelSampler,
                                            PatchPixelSampler, PixelSampler)
from nerfstudio.data.utils.dataloaders import (CacheDataloader,
                                               FixedIndicesEvalDataloader,
                                               RandIndicesEvalDataloader)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ParallelDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: ParallelDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
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
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    n_procs: int = 1
    """Number of processes to use for data loading."""

class DataProc(mp.Process):
    def __init__(
        self,
        out_queue: mp.Queue,
        config: ParallelDataManagerConfig,
        dataparser_outputs: DataparserOutputs,
        train_dataset: TDataset,
        pix_sampler: PixelSampler,
    ):
        super().__init__()
        self.daemon = True
        self.out_queue = out_queue
        self.config = config
        self.dataparser_outputs = dataparser_outputs
        self.train_dataset = train_dataset
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.dataparser_outputs is not None:
            cameras = self.dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        self.train_pixel_sampler = pix_sampler
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras)
        self.cache_images()
    
    def run(self):
        while True:
            batch = self.train_pixel_sampler.sample(self.img_data)
            ray_indices = batch["indices"]
            ray_bundle:RayBundle = self.train_ray_generator(ray_indices)
            ray_bundle = ray_bundle.pin_memory()
            self.out_queue.put((ray_bundle,batch), block=True)

    def cache_images(self):
        # caches all the input images in a NxHxWx3 tensor
        indices = range(len(self.train_dataset))
        batch_list = []
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for idx in indices:
                res = executor.submit(self.train_dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())
        self.img_data = self.config.collate_fn(batch_list)


class ParallelDataManager(DataManager, Generic[TDataset]):
    def __init__(
        self,
        config: ParallelDataManagerConfig,
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
        # self.sampler = None
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
        # self.eval_dataset = self.create_eval_dataset()
        self.pix_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        mp.set_start_method('spawn')
        self.data_q = mp.Queue(maxsize=4)
        self.data_procs = [
            DataProc(self.data_q, self.config, self.train_dataparser_outputs, self.train_dataset, self.pix_sampler)
            for i in range(self.config.n_procs)
        ]
        for proc in self.data_procs:
            proc.start()
        super().__init__()
        # Prime the executor with the first batch
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.batch_fut = self.executor.submit(self.data_q.get)

    def create_train_dataset(self) -> TDataset:
        """Sets up the data loaders for training"""
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, *args: Any, **kwargs: Any) -> PixelSampler:
        """Infer pixel sampler to use."""
        if self.config.patch_size > 1:
            return PatchPixelSampler(*args, **kwargs, patch_size=self.config.patch_size)

        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)
    
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        self.train_count += 1
        # Fetch the next batch in an executor to parallelize the queue get() operation
        # with the train step
        bundle, batch = self.batch_fut.result()
        self.batch_fut = self.executor.submit(self.data_q.get)
        bundle = bundle.to(self.device)
        return bundle,batch
    
    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
