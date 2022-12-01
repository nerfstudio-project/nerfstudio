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
Data manager for dreamfusion
"""

from __future__ import annotations

import random
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtyping import TensorType
from typing_extensions import Literal

import nerfstudio.utils.profiler as profiler
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper

CONSOLE = Console(width=120)


class TrivialDataset(InputDataset):
    """A trivial dataset with blank images for the viewer"""

    def __init__(self, cameras: Cameras):
        self.size = cameras.size
        self.cameras = cameras
        self.alpha_color = None
        self.scene_box = SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]]))
        self.mask_filenames = None
        self.metadata = to_immutable_dict({})

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Dict:
        return {"image": torch.cat([torch.ones(128, 256, 3), torch.zeros(128, 256, 3)], dim=0), "image_idx": index}


def random_train_pose(
    size,
    resolution,
    device,
    radius_mean=1.0,
    radius_std=0.1,
    central_rotation_range=[0, 360],
    vertical_rotation_range=[-90, 10],
    focal_range=[0.75, 1.35],
    jitter_std=0.01,
):
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        central_rotation_range: amount that we rotate around the center of the object
        vertical_rotation_range: amount that we allow the cameras to pan up and down from horizontal
    Return:
        poses: [size, 4, 4]
    """

    vertical_rotation_range = [vertical_rotation_range[0] + 90, vertical_rotation_range[1] + 90]
    # This is the uniform sample on the part of the sphere we care about where 0 = 0 degrees and 1 = 360 degrees
    sampled_uniform = (
        torch.rand(size) * (vertical_rotation_range[1] - vertical_rotation_range[0]) + vertical_rotation_range[0]
    ) / 180
    vertical_rotation = torch.arccos(1 - 2 * sampled_uniform)
    central_rotation = torch.deg2rad(
        torch.rand(size) * (central_rotation_range[1] - central_rotation_range[0]) + central_rotation_range[0]
    )

    rot_z = torch.stack(
        [
            torch.cos(central_rotation),
            -torch.sin(central_rotation),
            torch.zeros_like(central_rotation),
            torch.sin(central_rotation),
            torch.cos(central_rotation),
            torch.zeros_like(central_rotation),
            torch.zeros_like(central_rotation),
            torch.zeros_like(central_rotation),
            torch.ones_like(central_rotation),
        ],
        dim=-1,
    ).reshape(size, 3, 3)

    rot_x = torch.stack(
        [
            torch.ones_like(vertical_rotation),
            torch.zeros_like(vertical_rotation),
            torch.zeros_like(vertical_rotation),
            torch.zeros_like(vertical_rotation),
            torch.cos(vertical_rotation),
            -torch.sin(vertical_rotation),
            torch.zeros_like(vertical_rotation),
            torch.sin(vertical_rotation),
            torch.cos(vertical_rotation),
        ],
        dim=-1,
    ).reshape(size, 3, 3)

    # Default directions are facing in the -z direction, so origins should face opposite way
    origins = torch.stack([torch.tensor([0, 0, 1])] * size, dim=0)
    origins = (origins * radius_mean) + (origins * (torch.randn((origins.shape)) * radius_std))
    R = torch.bmm(rot_z, rot_x)  # Want to have Rx @ Ry @ origin
    print(torch.bmm(R, origins.unsqueeze(-1)).shape, (torch.randn((size, 3, 1)) * jitter_std).shape)
    t = torch.bmm(R, origins.unsqueeze(-1)) + torch.randn((size, 3, 1)) * jitter_std
    camera_to_worlds = torch.cat([R, t], dim=-1)

    focals = torch.rand(size) * (focal_range[1] - focal_range[0]) + focal_range[0]

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=focals * resolution,
        fy=focals * resolution,
        cx=resolution / 2,
        cy=resolution / 2,
    ).to(device)

    return cameras, torch.rad2deg(vertical_rotation), torch.rad2deg(central_rotation)


@dataclass
class DreamFusionDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DreamFusionDataManager)
    train_resolution: int = 64
    """Training resolution"""
    eval_resolution: int = 256
    """Evaluation resolution"""
    num_eval_angles: int = 256
    """Number of evaluation angles"""
    train_images_per_batch: int = 1
    """Number of images per batch for training"""
    eval_images_per_batch: int = 1
    """Number of images per batch for evaluation"""
    prompt: str = "A surfer sitting on a surfboard in the sky watching the sunset."
    """Prompt to optimize for."""


class DreamFusionDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DreamFusionDataManagerConfig

    def __init__(
        self,
        config: DreamFusionDataManagerConfig,
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
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser = self.config.dataparser.setup()

        self.eval_cameras, _, _ = random_train_pose(
            self.config.num_eval_angles,
            self.config.eval_resolution,
            device=device,
            # radius_mean=1,
            # radius_std=0.1,
            # central_rotation_range=[180, 180],
            # vertical_rotation_range=[0, 0],
            # focal_range=[1, 1],
            # jitter_std=0.1,
        )

        self.train_dataset = TrivialDataset(self.eval_cameras)
        self.eval_dataset = TrivialDataset(self.eval_cameras)

        DataManager.__init__(self)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        self.train_count += 1

        if step > 2000:
            cameras, _, _ = random_train_pose(
                self.config.train_images_per_batch, self.config.train_resolution, device=self.device
            )

            ray_bundle = cameras.generate_rays(torch.tensor([[i] for i in range(self.config.train_images_per_batch)]))
            return ray_bundle, {"initialization": False}

        cameras, vertical_rotation, central_rotation = random_train_pose(
            self.config.train_images_per_batch,
            self.config.train_resolution,
            device=self.device,
            radius_mean=1.5,
            radius_std=0,
            focal_range=[1, 1],
            vertical_rotation_range=[-180, 180],
            jitter_std=0,
        )
        ray_bundle = cameras.generate_rays(torch.tensor([[i] for i in range(self.config.train_images_per_batch)]))

        return ray_bundle, {"vertical": vertical_rotation, "central": central_rotation, "initialization": True}

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1

        if step > 2000:
            cameras, _, _ = random_train_pose(
                self.config.train_images_per_batch, self.config.train_resolution, device=self.device
            )

            ray_bundle = cameras.generate_rays(torch.tensor([[i] for i in range(self.config.eval_images_per_batch)]))
            return ray_bundle, {"initialization": False}

        cameras, vertical_rotation, central_rotation = random_train_pose(
            self.config.eval_images_per_batch,
            self.config.eval_resolution,
            device=self.device,
            radius_mean=1.5,
            radius_std=0,
            focal_range=[1, 1],
            vertical_rotation_range=[-180, 180],
            jitter_std=0,
        )
        ray_bundle = cameras.generate_rays(torch.tensor([[i] for i in range(self.config.train_images_per_batch)]))

        return ray_bundle, {"vertical": vertical_rotation, "central": central_rotation}

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        raise ValueError("No more eval images")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups
