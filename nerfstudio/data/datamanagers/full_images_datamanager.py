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

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import cv2
import fpsample
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter
from typing_extensions import assert_never

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class FullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
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
    cache_images: Literal["cpu", "gpu", "disk"] = "gpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device. If "disk", keeps images on disk. """
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    """Specifies which sampling strategy is used to generate train cameras, 'random' means sampling 
    uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the artifacts 
    due to oversampling subsets of cameras that are very close to each other."""
    train_cameras_sampling_seed: int = 42
    """Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across 
    different runs."""
    fps_reset_every: int = 100
    """The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every
    samples from the pool of all training cameras without replacement before a new round of sampling starts."""
    use_image_train_dataloader: bool = cache_images == "disk"
    """Supports datasets that do not fit in system RAM and allows parallelization of the dataloading process with multiple workers."""
    dataloader_num_workers: int = 0
    """The number of workers performing the dataloading from either disk/RAM, which 
    includes collating, pixel sampling, unprojecting, ray generation etc."""
    prefetch_factor: int = 1
    """The limit number of batches a worker will start loading once an iterator is created. 
    More details are described here: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""

class FullImageDatamanager(DataManager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
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
        # print(type(self.train_dataset)) # prints InputDataset
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

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

    @cached_property
    def cached_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("train", cache_images_device=self.config.cache_images)

    @cached_property
    def cached_eval(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_images("eval", cache_images_device=self.config.cache_images)

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)
        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
                f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
                f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            )
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)

        return undistorted_images

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

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval)
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

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
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(0)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()
        data = self.cached_train[image_idx]
        data["image"] = data["image"].to(self.device)

        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data


def _undistort_image(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        assert distortion_params[3] == 0, (
            "We don't support the 4th Brown parameter for image undistortion, "
            "Only k1, k2, k3, p1, p2 can be non-zero."
        )
        #print(distortion_params) # [ 0.05517609 -0.07427584  0.          0.         -0.00026702 -0.00060216]
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
            breakpoint()
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
            # print("1:", image.shape) # prints (960, 540, 3)
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        # print(x, y, w, h) # prints 0, 0, 539, 959
        # image = image[y : y + h, x : x + w]
        # print("2:", image.shape) # prints (959, 539, 3)

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
                borderValue=0,
            )
            / 255.0
        ).bool()[..., None]
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        assert mask.shape == (undist_h, undist_w, 1)
        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")
    # print("final:", image.shape, camera.width, camera.height) # prints 'final: (959, 539, 3) tensor([540]) tensor([960])'
    return K, image, mask


## Let's implement a parallelized splat dataloader!
def undistort_idx(idx: int, dataset: TDataset, image_type: Literal["uint8", "float32"] = "float32") -> Dict[str, torch.Tensor]:
    """Undistorts an image to one taken by a linear (pinhole) camera model and updates the dataset's camera intrinsics to pinhole"""
    data = dataset.get_data(idx, image_type)
    camera = dataset.cameras[idx].reshape(())
    # dataset.cameras.width[idx] = data["image"].shape[1]
    # dataset.cameras.height[idx] = data["image"].shape[0]
    if idx == 48:
        print("beginning", camera.width, camera.height)
    assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
        f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
        f'does not match the camera parameters ({camera.width.item(), camera.height.item()}), idx = {idx}'
    )
    if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
        return data
    K = camera.get_intrinsics_matrices().numpy()
    distortion_params = camera.distortion_params.numpy()
    image = data["image"].numpy()
    K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
    # print(image.shape[1]) # outputs 539
    # print(cameras[48].reshape(()).width.item()) # outputs 540
    data["image"] = torch.from_numpy(image)
    if mask is not None:
        data["mask"] = mask

    dataset.cameras.fx[idx] = float(K[0, 0])
    dataset.cameras.fy[idx] = float(K[1, 1])
    dataset.cameras.cx[idx] = float(K[0, 2])
    dataset.cameras.cy[idx] = float(K[1, 2])
    dataset.cameras.width[idx] = image.shape[1]
    dataset.cameras.height[idx] = image.shape[0]
    if idx == 48:
        print("ending", camera.width, camera.height)
    return data

import math
class ImageBatchStream(torch.utils.data.IterableDataset):
    """
    A wrapper of InputDataset that outputs undistorted full images and cameras. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    def __init__(
        self,
        datamanager_config: DataManagerConfig,
        input_dataset: TDataset,
        device,
    ):
        self.config = datamanager_config
        self.input_dataset = input_dataset
        self.device = device

    def __iter__(self):
        # print(self.input_dataset.cameras.device) prints cpu
        dataset_indices = list(
            range(len(self.input_dataset))
        )
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # if we have multiple processes
            per_worker = int(math.ceil(len(dataset_indices) / float(worker_info.num_workers)))
            slice_start = worker_info.id * per_worker
        else: # we only have a single process
            per_worker = len(self.input_dataset)
            slice_start = 0
        worker_indices = dataset_indices[
            slice_start : slice_start + per_worker
        ]  # the indices of the datapoints in the dataset this worker will load
        r = random.Random(self.config.train_cameras_sampling_seed)
        r.shuffle(worker_indices)
        i = 0 # i refers to what image index we are outputting: i=0 => we are yielding our first image,camera
        while True:
            if i % per_worker == 0: # if we've iterated through all the worker's partition of images, we need to reshuffle
                r.shuffle(worker_indices)
                i = 0
            idx = worker_indices[i] # idx refers to the actual datapoint index this worker will retrieve
            data = undistort_idx(idx, self.input_dataset, self.config.cache_images_type)
            camera = self.input_dataset.cameras[idx : idx + 1]#.to(self.device)
            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = idx
            i += 1
            if torch.sum(camera.camera_to_worlds) == 0:
                print(i, camera.camera_to_worlds, "YOYO INSIDE IMAGEBATCHSTREAM")
            yield camera, data

class ParallelFullImageDatamanager(FullImageDatamanager, Generic[TDataset]):
    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs
    ):
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs
        )
 
    def setup_train(self):
        self.train_imagebatch_stream = ImageBatchStream(
            input_dataset=self.train_dataset,
            datamanager_config=self.config,
            device=self.device,
        )
        self.train_image_dataloader = torch.utils.data.DataLoader(
            self.train_imagebatch_stream,
            batch_size=1,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=identity_collate,
            pin_memory_device=self.device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def setup_eval(self):
        self.eval_imagebatch_stream = ImageBatchStream(
            input_dataset=self.eval_dataset,
            datamanager_config=self.config,
            device=self.device,
        )
        self.eval_image_dataloader = torch.utils.data.DataLoader(
            self.eval_imagebatch_stream,
            batch_size=1,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=identity_collate,
            pin_memory_device=self.device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader) # these things output tuples

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        return self.iter_eval_image_dataloader

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        self.train_count += 1
        camera, data = next(self.iter_train_image_dataloader)[0]
        return camera, data
    
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        self.eval_count += 1
        camera, data = next(self.iter_train_image_dataloader)[0]
        return camera, data
    