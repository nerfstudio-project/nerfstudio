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
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import math
import multiprocessing
import random
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Sized, Tuple, Union, cast

import cv2
import numpy as np
import torch
from rich.progress import track
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE


def variable_res_collate(batch: List[Dict]) -> Dict:
    """Default collate function for our dataloader.
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


def _undistort_image(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        assert distortion_params[3] == 0, (
            "We don't support the 4th Brown parameter for image undistortion, Only k1, k2, k3, p1, p2 can be non-zero."
        )
        # we rearrange the distortion parameters because OpenCV expects the order (k1, k2, p1, p2, k3)
        # see https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        # because OpenCV expects the pixel coord to be top-left, we need to shift the principal point by 0.5
        # see https://github.com/nerfstudio-project/nerfstudio/issues/3048
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        newK[0, 2] -= x
        newK[1, 2] -= y

        if "depth_image" in data:
            data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
            mask = mask[y : y + h, x : x + w]
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK

    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        K[0, 2] = K[0, 2] - 0.5
        K[1, 2] = K[1, 2] - 0.5
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
        )
        # and then remap:
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] = newK[0, 2] + 0.5
        newK[1, 2] = newK[1, 2] + 0.5
        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        fisheye624_params = torch.cat(
            [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
        )
        assert fisheye624_params.shape == (16,)
        assert (
            "mask" not in data
            and camera.metadata is not None
            and "fisheye_crop_radius" in camera.metadata
            and isinstance(camera.metadata["fisheye_crop_radius"], float)
        )
        fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

        # Approximate the FOV of the unmasked region of the camera.
        upper, lower, left, right = fisheye624_unproject_helper(
            torch.tensor(
                [
                    [camera.cx, camera.cy - fisheye_crop_radius],
                    [camera.cx, camera.cy + fisheye_crop_radius],
                    [camera.cx - fisheye_crop_radius, camera.cy],
                    [camera.cx + fisheye_crop_radius, camera.cy],
                ],
                dtype=torch.float32,
            )[None],
            params=fisheye624_params[None],
        ).squeeze(dim=0)
        fov_radians = torch.max(
            torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
            torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
        )

        # Heuristics to determine parameters of an undistorted image.
        undist_h = int(fisheye_crop_radius * 2)
        undist_w = int(fisheye_crop_radius * 2)
        undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
        undist_K = torch.eye(3)
        undist_K[0, 0] = undistort_focal  # fx
        undist_K[1, 1] = undistort_focal  # fy
        undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
        undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

        # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
        undist_uv_homog = torch.stack(
            [
                *torch.meshgrid(
                    torch.arange(undist_w, dtype=torch.float32),
                    torch.arange(undist_h, dtype=torch.float32),
                ),
                torch.ones((undist_w, undist_h), dtype=torch.float32),
            ],
            dim=-1,
        )
        assert undist_uv_homog.shape == (undist_w, undist_h, 3)
        dist_uv = (
            fisheye624_project(
                xyz=(
                    torch.einsum(
                        "ij,bj->bi",
                        torch.linalg.inv(undist_K),
                        undist_uv_homog.reshape((undist_w * undist_h, 3)),
                    )[None]
                ),
                params=fisheye624_params[None, :],
            )
            .reshape((undist_w, undist_h, 2))
            .numpy()
        )
        map1 = dist_uv[..., 1]
        map2 = dist_uv[..., 0]

        # Use correspondence to undistort image.
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Compute undistorted mask as well.
        dist_h = camera.height.item()
        dist_w = camera.width.item()
        mask = np.mgrid[:dist_h, :dist_w]
        mask[0, ...] -= dist_h // 2
        mask[1, ...] -= dist_w // 2
        mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
        mask = torch.from_numpy(
            cv2.remap(
                mask.astype(np.uint8) * 255,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            / 255.0
        ).bool()[..., None]
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        assert mask.shape == (undist_h, undist_w, 1)
        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")
    return K, image, mask


def undistort_view(
    idx: int, dataset: InputDataset, image_type: Literal["uint8", "float32"] = "float32"
) -> Tuple[Cameras, Dict]:
    """Undistorts an image to one taken by a linear (pinhole) camera model and returns a new Camera with these updated intrinsics
    Note: this method does not modify the dataset's attributes at all.

    Returns: The undistorted data (image, depth, mask, etc.) and the new linear Camera object
    """
    data = dataset.get_data(idx, image_type)
    camera = dataset.cameras[idx].reshape(())
    assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
        f"The size of image ({data['image'].shape[1]}, {data['image'].shape[0]}) loaded "
        f"does not match the camera parameters ({camera.width.item(), camera.height.item()}), idx = {idx}"
    )
    if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
        return camera.reshape((1,)), data
    K = camera.get_intrinsics_matrices().numpy()
    distortion_params = camera.distortion_params.numpy()
    image = data["image"].numpy()
    K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
    data["image"] = torch.from_numpy(image)
    if mask is not None:
        data["mask"] = mask

    # create a new Camera with the rectified / undistorted intrinsics
    new_camera = Cameras(
        camera_to_worlds=camera.camera_to_worlds.unsqueeze(0),
        fx=torch.Tensor([[float(K[0, 0])]]),
        fy=torch.Tensor([[float(K[1, 1])]]),
        cx=torch.Tensor([[float(K[0, 2])]]),
        cy=torch.Tensor([[float(K[1, 2])]]),
        width=torch.Tensor([[image.shape[1]]]).to(torch.int32),
        height=torch.Tensor([[image.shape[0]]]).to(torch.int32),
    )
    return new_camera, data


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 or infinity for all images.
        num_times_to_repeat_images: How often to yield an image batch before resampling. -1 or infinity to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_sample_from: Union[int, float] = float("inf"),
        num_times_to_repeat_images: Union[int, float] = float("inf"),
        device: Union[torch.device, str] = "cpu",
        collate_fn: Callable[[Any], Any] = nerfstudio_collate,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.num_times_to_repeat_images = num_times_to_repeat_images
        self.cache_all_images = (num_images_to_sample_from == -1) or (num_images_to_sample_from >= len(self.dataset))
        self.num_images_to_sample_from = len(self.dataset) if self.cache_all_images else num_images_to_sample_from
        self.device = device
        self.collate_fn = collate_fn
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device

        self.num_repeated = self.num_times_to_repeat_images  # starting value
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            CONSOLE.print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                CONSOLE.print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_times_to_repeat_images == float("inf"):
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, without resampling."
            )
        else:
            CONSOLE.print(
                f"Caching {self.num_images_to_sample_from} out of {len(self.dataset)} images, "
                f"resampling every {self.num_times_to_repeat_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        indices = random.sample(
            range(len(self.dataset)),
            k=(
                len(self.dataset)
                if self.num_images_to_sample_from == float("inf")
                else int(self.num_images_to_sample_from)
            ),
        )
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch of images."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_times_to_repeat_images != -1 and self.num_repeated >= self.num_times_to_repeat_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_times_to_repeat_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


class RayBatchStream(IterableDataset):
    """Wrapper around Pytorch's IterableDataset to generate the next batch of rays (next RayBundle) and corresponding labels
    with multiple parallel workers.

    Each worker samples a small batch of images, pixel samples those images, and generates rays for one training step.
    The same batch of images can be pixel sampled multiple times hasten ray generation, as retrieving images is process
    bottlenecked by disk read speed. To avoid Out-Of-Memory (OOM) errors, this batch of images is small and regenerated
    by resampling the worker's partition of images to maintain sampling diversity.
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        sampling_seed: int = 3301,
        num_rays_per_batch: int = 1024,
        num_images_to_sample_from: Union[int, float] = float("inf"),
        num_times_to_repeat_images: Union[int, float] = float("inf"),
        device: Union[torch.device, str] = "cpu",
        # variable_res_collate avoids np.stack'ing images, which allows it to be much faster than `nerfstudio_collate`
        collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(variable_res_collate)),
        num_image_load_threads: int = 4,
        exclude_batch_keys_from_device: Optional[List[str]] = None,
        load_from_disk: bool = False,
        patch_size: int = 1,
        custom_ray_processor: Optional[Callable[[RayBundle, Dict], Tuple[RayBundle, Dict]]] = None,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.input_dataset = input_dataset
        self.sampling_seed = sampling_seed
        assert isinstance(self.input_dataset, Sized)
        self.num_rays_per_batch = num_rays_per_batch
        """Number of rays per batch to user per training iteration."""
        self.num_images_to_sample_from = num_images_to_sample_from
        """How many images to sample to generate a RayBundle. More images means greater sampling diversity at expense of increased RAM usage."""
        self.num_times_to_repeat_images = num_times_to_repeat_images
        """How many RayBundles to generate from this batch of images after sampling `num_images_to_sample_from` images."""
        self.device = device
        """If a CUDA GPU is present, self.device will be set to use that GPU."""
        self.collate_fn = collate_fn
        """What collate function is used to batch images to be used for pixel sampling and ray generation. """
        self.num_image_load_threads = num_image_load_threads
        """Number of threads created to read images from disk and form collated batches."""
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device
        """Which key of the batch (such as 'image', 'mask','depth') to prevent from moving to the device. 
        For instance, if you would like to conserve GPU memory, don't move the image tensors to the GPU, 
        which comes at a cost of total training time. The default value is ['image']."""
        self.load_from_disk = load_from_disk
        """If True, conserves RAM memory by loading images from disk.
        If False, each worker caches all the images in its dataset partition as tensors to RAM and loads from RAM."""
        self.patch_size = patch_size
        """Size of patch to sample from. If > 1, patch-based sampling will be used."""
        self._cached_collated_batch = None
        """Each worker has a self._cached_collated_batch contains a collated batch of images cached in RAM for a specific worker that's ready for pixel sampling."""
        self.pixel_sampler_config: PixelSamplerConfig = PixelSamplerConfig()
        """Specifies the pixel sampler config used to sample pixels from images. Each worker will have its own pixel sampler"""
        self.ray_generator: Optional[RayGenerator] = None
        """Each worker will have its own ray generator, so this is set to None for now."""
        self.custom_ray_processor = custom_ray_processor

    def _get_pixel_sampler(self, dataset: InputDataset, num_rays_per_batch: int) -> PixelSampler:
        """copied from VanillaDataManager."""
        from nerfstudio.cameras.cameras import CameraType

        if self.patch_size > 1 and type(self.pixel_sampler_config) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(patch_size=self.patch_size, num_rays_per_batch=num_rays_per_batch)
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")

        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")

        return self.pixel_sampler_config.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def _get_batch_list(self, indices):
        """Returns a list representing a single batch from the dataset attribute.
        Each item of the list is a dictionary with dict_keys(['image_idx', 'image']) representing 1 image.
        This function is used to sample and load images from disk/RAM and is only called in _get_collated_batch()
        The length of the list is equal to the (# of training images) / (num_workers)

        Note: The `indices` given to _get_collated_batch() are the `indices` passed to _get_batch_list(). These `indices`
        are either set to the entire dataset if we are not loading from disk or some partiton of dataset whose size
        is dependent on self.num_images_to_sample_from if we are loading from disk.
        """
        assert isinstance(self.input_dataset, Sized)
        batch_list = []
        results = []

        num_threads = int(self.num_image_load_threads) if self.load_from_disk else 4 * int(self.num_image_load_threads)
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        # NB: this is I/O heavy because we are going to disk and reading an image filename
        # hence multi-threaded inside the worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.input_dataset.__getitem__, idx)
                results.append(res)
            for res in results:
                batch_list.append(res.result())

        return batch_list

    def _get_collated_batch(self, indices):
        """Takes the output of _get_batch_list and collates them with nerfstudio_collate() or variable_res_collate()
        Note: dict is an instance of collections.abc.Mapping

        The resulting output is collated_batch: a dictionary with dict_keys(['image_idx', 'image'])
            - collated_batch['image_idx'] is tensor with shape torch.Size([per_worker])
            - collated_batch['image'] is tensor with shape torch.Size([per_worker, height, width, 3])
        """
        batch_list = self._get_batch_list(indices=indices)
        collated_batch = self.collate_fn(batch_list)
        collated_batch = get_dict_to_torch(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        """This implementation allows every worker only cache the indices of the images they will use to generate rays to conserve RAM memory."""
        worker_info = get_worker_info()
        if worker_info is not None:  # if we have multiple processes
            if len(self.input_dataset) < worker_info.num_workers:
                # if there's fewer datapoints than workers, each worker receives all datapoints
                worker_indices = list(range(len(self.input_dataset)))
            else:
                per_worker = int(math.ceil(len(self.input_dataset) / float(worker_info.num_workers)))
                slice_start = worker_info.id * per_worker
                dataset_indices = list(range(len(self.input_dataset)))
                worker_indices = dataset_indices[slice_start : slice_start + per_worker]
        else:  # if we only have a single process
            worker_indices = list(range(len(self.input_dataset)))
        if not self.load_from_disk:
            self._cached_collated_batch = self._get_collated_batch(worker_indices)
        r = random.Random(self.sampling_seed)
        num_rays_per_loop = self.num_rays_per_batch  # default train_num_rays_per_batch is 4096

        # each worker has its own pixel sampler
        worker_pixel_sampler = self._get_pixel_sampler(self.input_dataset, num_rays_per_loop)

        # the generated RayBundles will be on the same device as self.input_dataset.cameras (CPU)
        self.ray_generator = RayGenerator(self.input_dataset.cameras)

        i = 0
        true_random = random.Random(worker_info.id) if worker_info is not None else r
        # We offset the value of repeat so that they're not all running out of images at once
        repeat_offset_max = 10 if worker_info is not None else 1
        repeat_offset = true_random.randint(0, repeat_offset_max)
        while True:
            if not self.load_from_disk:
                collated_batch = self._cached_collated_batch
            elif i % (self.num_times_to_repeat_images + repeat_offset) == 0:
                r.shuffle(worker_indices)
                repeat_offset = true_random.randint(0, repeat_offset_max)
                if self.num_images_to_sample_from == float("inf"):
                    # if infinity, the worker gets all available indices in its partition
                    image_indices = worker_indices
                else:
                    # get a total of 'num_images_to_sample_from' image indices
                    image_indices = worker_indices[: self.num_images_to_sample_from]

                collated_batch = self._get_collated_batch(image_indices)
            i += 1
            """
            Here, the variable 'batch' refers to the output of our pixel sampler.
                - batch is a dict_keys(['image', 'indices'])
                - batch['image'] returns a pytorch tensor with shape `torch.Size([4096, 3])` , where 4096 = num_rays_per_batch. 
                    - Note: each row in this tensor represents the RGB values as floats in [0, 1] of the pixel the ray goes through. 
                    - The info of what specific image index that pixel belongs to is stored within batch[’indices’]
                - batch['indices'] returns a pytorch tensor `torch.Size([4096, 3])` tensor where each row represents (image_idx, pixelRow, pixelCol)
            pixel_sampler (for variable_res_collate) will loop though each image, samples pixel within the mask, and returns 
            them as the variable `indices` which has shape torch.Size([4096, 3]), where each row represents a pixel (image_idx, pixelRow, pixelCol)
            """
            batch = worker_pixel_sampler.sample(collated_batch)  # type: ignore
            # Note: collated_batch["image"].get_device() will return CPU if self.exclude_batch_keys_from_device contains 'image'
            ray_indices = batch["indices"]
            # the ray_bundle is on the GPU; batch["image"] is on the CPU, here we move it to the GPU
            ray_bundle = self.ray_generator(ray_indices).to(self.device)
            if self.custom_ray_processor:
                ray_bundle, batch = self.custom_ray_processor(ray_bundle, batch)

            yield ray_bundle, batch


class ImageBatchStream(IterableDataset):
    """
    A wrapper of InputDataset that outputs undistorted full images and cameras. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        sampling_seed: int = 3301,
        cache_images_type: Literal["uint8", "float32"] = "float32",
        device: Union[torch.device, str] = "cpu",
        custom_image_processor: Optional[Callable[[Cameras, Dict], Tuple[Cameras, Dict]]] = None,
    ):
        self.input_dataset = input_dataset
        self.sampling_seed = sampling_seed
        self.cache_images_type = cache_images_type
        self.device = device
        self.custom_image_processor = custom_image_processor

    def __iter__(self):
        dataset_indices = list(range(len(self.input_dataset)))
        worker_info = get_worker_info()
        if worker_info is not None:  # if we have multiple processes
            per_worker = int(math.ceil(len(dataset_indices) / float(worker_info.num_workers)))
            slice_start = worker_info.id * per_worker
        else:  # we only have a single process
            per_worker = len(self.input_dataset)
            slice_start = 0
        worker_indices = dataset_indices[
            slice_start : slice_start + per_worker
        ]  # the indices of the datapoints in the dataset this worker will load
        r = random.Random(self.sampling_seed)
        r.shuffle(worker_indices)
        i = 0  # i refers to what image index we are outputting: i=0 => we are yielding our first image,camera

        while True:
            if i >= len(worker_indices):
                # if we've iterated through all the worker's partition of images, we need to reshuffle
                r.shuffle(worker_indices)
                i = 0
            idx = worker_indices[i]  # idx refers to the actual datapoint index this worker will retrieve
            camera, data = undistort_view(idx, self.input_dataset, self.cache_images_type)  # type: ignore
            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = idx

            # Apply custom processing if provided
            if self.custom_image_processor:
                camera, data = self.custom_image_processor(camera, data)

            i += 1
            camera = camera.to(self.device)
            for k in data.keys():
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)
            yield camera, data


class EvalDataloader(DataLoader):
    """Evaluation dataloader base class

    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        self.input_dataset = input_dataset
        self.cameras = input_dataset.cameras.to(device)
        self.device = device
        self.kwargs = kwargs
        super().__init__(dataset=input_dataset)

    @abstractmethod
    def __iter__(self):
        """Iterates over the dataset"""
        return self

    @abstractmethod
    def __next__(self) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data"""

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Get camera for the given image index

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return camera, batch

    def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return ray_bundle, batch


class FixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns a fixed set of indices.

    Args:
        input_dataset: InputDataset to load data from
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            camera, batch = self.get_camera(image_idx)
            self.count += 1
            return camera, batch
        raise StopIteration


class RandIndicesEvalDataloader(EvalDataloader):
    """Dataloader that returns random images.
    Args:
        input_dataset: InputDataset to load data from
        device: Device to load data to
    """

    def __iter__(self):
        return self

    def __next__(self):
        # choose a random image index
        image_idx = random.randint(0, len(self.cameras) - 1)
        camera, batch = self.get_camera(image_idx)
        return camera, batch
