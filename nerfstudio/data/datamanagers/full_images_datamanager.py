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
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import cv2
import numpy as np
import torch
from copy import deepcopy
from torch.nn import Parameter
from tqdm import tqdm

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class FullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    dataparser: AnnotatedDataParserUnion = NerfstudioDataParserConfig()
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
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""


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
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print("Train dataset has over 500 images, overriding cach_images to cpu", style="bold yellow")
            self.config.cache_images = "cpu"
        self.cached_train, self.cached_eval = self.cache_images(self.config.cache_images)
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

    def cache_images(self, cache_images_option):
        cached_train = []
        cached_eval = []
        CONSOLE.log("Caching / undistorting train images")
        for i in tqdm(range(len(self.train_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.train_dataset.get_data(i)
            camera = self.train_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                continue
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()
            if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
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
                if np.any(distortion_params):
                    newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                    image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
                else:
                    newK = K
                    roi = 0, 0, image.shape[1], image.shape[0]
                # crop the image and update the intrinsics accordingly
                x, y, w, h = roi
                image = image[y : y + h, x : x + w]
                if "depth_image" in data:
                    data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
                # update the width, height
                self.train_dataset.cameras.width[i] = w
                self.train_dataset.cameras.height[i] = h
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    if np.any(distortion_params):
                        mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
                    mask = mask[y : y + h, x : x + w]
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK

            elif camera.camera_type.item() == CameraType.FISHEYE.value:
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
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK
            else:
                raise NotImplementedError("Only perspective and fisheye cameras are supported")
            data["image"] = torch.from_numpy(image)

            cached_train.append(data)

            self.train_dataset.cameras.fx[i] = float(K[0, 0])
            self.train_dataset.cameras.fy[i] = float(K[1, 1])
            self.train_dataset.cameras.cx[i] = float(K[0, 2])
            self.train_dataset.cameras.cy[i] = float(K[1, 2])

        CONSOLE.log("Caching / undistorting eval images")
        for i in tqdm(range(len(self.eval_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.eval_dataset.get_data(i)
            camera = self.eval_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                continue
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()
            if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
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
                if np.any(distortion_params):
                    newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
                    image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
                else:
                    newK = K
                    roi = 0, 0, image.shape[1], image.shape[0]
                # crop the image and update the intrinsics accordingly
                x, y, w, h = roi
                image = image[y : y + h, x : x + w]
                # update the width, height
                self.eval_dataset.cameras.width[i] = w
                self.eval_dataset.cameras.height[i] = h
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    if np.any(distortion_params):
                        mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
                    mask = mask[y : y + h, x : x + w]
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK

            elif camera.camera_type.item() == CameraType.FISHEYE.value:
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
                image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                if "mask" in data:
                    mask = data["mask"].numpy()
                    mask = mask.astype(np.uint8) * 255
                    mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
                    data["mask"] = torch.from_numpy(mask).bool()
                K = newK
            else:
                raise NotImplementedError("Only perspective and fisheye cameras are supported")
            data["image"] = torch.from_numpy(image)

            cached_eval.append(data)

            self.eval_dataset.cameras.fx[i] = float(K[0, 0])
            self.eval_dataset.cameras.fy[i] = float(K[1, 1])
            self.eval_dataset.cameras.cx[i] = float(K[0, 2])
            self.eval_dataset.cameras.cy[i] = float(K[1, 2])

        if cache_images_option == "gpu":
            for cache in cached_train:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
            for cache in cached_eval:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        else:
            for cache in cached_train:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
            for cache in cached_eval:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()

        return cached_train, cached_eval

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
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

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
