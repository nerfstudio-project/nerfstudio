from __future__ import annotations

import concurrent.futures
import functools
import time
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, Tuple, Type, Union, cast

import torch
import torch.multiprocessing as mp
from rich.progress import track
from torch.nn import Parameter

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.lie_groups import exp_map_SE3
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset, variable_res_collate
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PatchPixelSampler, PixelSampler
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import poses as pose_utils
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader, CacheDataloader, FixedIndicesEvalDataloader


@dataclass
class ParallelDataManagerConfig(DataManagerConfig):
    """Config for a `ParallelDataManager` which reads data in multiple processes"""

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
    """Number of processes to use for train data loading."""
    queue_size: int = 2
    """Size of shared data queue containing generated ray bundles and batches. 
    If queue_size <= 0, the queue size is infinite."""
    max_thread_workers: Optional[int] = None
    """Maximum number of threads to use in thread pool executor. If None, automatically
    set to harware cpu_count + 4."""
    position_noise_std: float = 0.0
    """Noise to add to initial camera pose positions. Useful for debugging."""
    orientation_noise_std: float = 0.0
    """Noise to add to initial camera pose orientations. Useful for debugging."""
    non_trainable_camera_indices: Optional[Tuple[int, ...]] = None
    """List of non trainable camera indices."""


class DataProcessor(mp.Process):
    """Parallel dataset batch processor.

    This class is responsible for generating ray bundles from an input dataset
    in parallel python processes.

    Args:
        out_queue: the output queue for storing the processed data
        config: configuration object for the parallel data manager
        dataparser_outputs: outputs from the dataparser
        dataset: input dataset
        pixel_sampler: The pixel sampler for sampling rays
    """

    def __init__(
        self,
        out_queue: mp.Queue,
        config: ParallelDataManagerConfig,
        dataparser_outputs: DataparserOutputs,
        dataset: TDataset,
        pixel_sampler: PixelSampler,
    ):
        super().__init__()
        self.daemon = True
        self.out_queue = out_queue
        self.config = config
        self.dataparser_outputs = dataparser_outputs
        self.dataset = dataset
        self.exclude_batch_keys_from_device = self.dataset.exclude_batch_keys_from_device
        self.pixel_sampler = pixel_sampler
        self.ray_generator = RayGenerator(self.dataset.cameras)
        self.cache_images()

    def run(self):
        """Append out queue in parallel with ray bundles and batches."""
        while True:
            batch = self.pixel_sampler.sample(self.img_data)
            ray_indices = batch["indices"]
            ray_bundle: RayBundle = self.ray_generator(ray_indices)
            ray_bundle = ray_bundle.pin_memory()
            while True:
                try:
                    self.out_queue.put_nowait((ray_bundle, batch))
                    break
                except queue.Full:
                    time.sleep(0.0001)
                except Exception:
                    CONSOLE.print_exception()
                    CONSOLE.print("[bold red]Error: Error occured in parallel datamanager queue.")

    def cache_images(self):
        # caches all the input images in a NxHxWx3 tensor
        indices = range(len(self.dataset))
        batch_list = []
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_thread_workers) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)
            for res in track(results, description="Loading data batch", transient=False):
                batch_list.append(res.result())
        self.img_data = self.config.collate_fn(batch_list)


class ParallelDataManager(DataManager, Generic[TDataset]):
    """Data manager implementation for parallel dataloading.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

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
        self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        cameras = self.train_dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break
        self.non_trainable_camera_indices = self.config.non_trainable_camera_indices
        if self.config.position_noise_std != 0.0 or self.config.orientation_noise_std != 0.0:
            self.apply_pose_noise(self.non_trainable_camera_indices)
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        # Spawn is critical for not freezing the program (PyTorch compatability issue)
        mp.set_start_method("spawn")
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

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        self.train_pix_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        # Manager().queue() can be faster
        # https://stackoverflow.com/questions/43439194/python-multiprocessing-queue-vs-multiprocessing-manager-queue/45236748#45236748
        self.data_queue = mp.Manager().Queue(maxsize=self.config.queue_size)
        # self.data_queue = mp.Queue(maxsize=self.config.queue_size)
        self.data_procs = [
            DataProcessor(
                out_queue=self.data_queue,
                config=self.config,
                dataparser_outputs=self.train_dataparser_outputs,
                dataset=self.train_dataset,
                pixel_sampler=self.train_pix_sampler,
            )
            for i in range(self.config.n_procs)
        ]
        for proc in self.data_procs:
            proc.start()

        # Prime the executor with the first batch
        self.train_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_thread_workers)
        self.train_batch_fut = self.train_executor.submit(self.data_queue.get)

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
        self.train_count += 1

        # Fetch the next batch in an executor to parallelize the queue get() operation
        # with the train step
        bundle, batch = self.train_batch_fut.result()
        self.train_batch_fut = self.train_executor.submit(self.data_queue.get)
        ray_bundle = bundle.to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data fnon_trainable_camera_indicesrom the eval dataloader."""
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

    def apply_pose_noise(self, non_trainable_camera_indices: Optional[List[int]]):
        """Apply noise to training camera poses.

        Args:
            non_trainable_camera_indices: list of non trainable cameras
        """
        assert self.config.position_noise_std >= 0.0 and self.config.orientation_noise_std >= 0.0
        camera_to_worlds = self.train_dataparser_outputs.cameras.camera_to_worlds.to(self.device)
        num_cameras = len(self.train_dataparser_outputs.cameras)
        std_vector = torch.tensor(
            [self.config.position_noise_std] * 3 + [self.config.orientation_noise_std] * 3, device=self.device
        )
        pose_noise = exp_map_SE3(torch.normal(torch.zeros((num_cameras, 6), device=self.device), std_vector))
        if non_trainable_camera_indices is not None:
            pose_noise[torch.tensor(non_trainable_camera_indices).long()] = torch.eye(4, device=pose_noise.device)[
                :3, :4
            ]
        self.train_dataparser_outputs.cameras.camera_to_worlds = functools.reduce(
            pose_utils.multiply, [camera_to_worlds, pose_noise]
        ).to(self.train_dataparser_outputs.cameras.camera_to_worlds.device)
