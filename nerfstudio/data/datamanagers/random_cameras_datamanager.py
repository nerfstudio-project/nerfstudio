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
Data manager without input images, only random camera poses
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union

import torch
from rich.progress import Console
from torch import Tensor
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader

CONSOLE = Console(width=120)


class TrivialDataset(InputDataset):
    """A trivial dataset with blank images for the viewer"""

    # pylint: disable=super-init-not-called
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
        return {
            "image": torch.cat([torch.ones(128, 256, 3), torch.zeros(128, 256, 3)], dim=0),
            "image_idx": index,
        }


def random_train_pose(
    size: int,
    resolution: int,
    device: Union[torch.device, str],
    radius_mean: float = 1.0,
    radius_std: float = 0.1,
    central_rotation_range: Tuple[float, float] = (0, 360),
    vertical_rotation_range: Tuple[float, float] = (-90, 0),
    focal_range: Tuple[float, float] = (0.75, 1.35),
    jitter_std: float = 0.01,
    center: Tuple[float, float, float] = (0, 0, 0),
) -> Tuple[Cameras, Tensor, Tensor]:
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_mean: mean radius of the orbit camera.
        radius_std: standard deviation of the radius of the orbit camera.
        central_rotation_range: amount that we rotate around the center of the object
        vertical_rotation_range: amount that we allow the cameras to pan up and down from horizontal
        focal_range: focal length range
        jitter_std: standard deviation of the jitter added to the camera position
        center: center of the object
    Return:
        poses: [size, 4, 4]
    """

    vertical_rotation_range = (
        vertical_rotation_range[0] + 90,
        vertical_rotation_range[1] + 90,
    )
    # This is the uniform sample on the part of the sphere we care about where 0 = 0 degrees and 1 = 360 degrees
    sampled_uniform = (
        torch.rand(size) * (vertical_rotation_range[1] - vertical_rotation_range[0]) + vertical_rotation_range[0]
    ) / 180
    vertical_rotation = torch.arccos(1 - 2 * sampled_uniform)
    central_rotation = torch.deg2rad(
        torch.rand(size) * (central_rotation_range[1] - central_rotation_range[0]) + central_rotation_range[0]
    )

    c_cos = torch.cos(central_rotation)
    c_sin = torch.sin(central_rotation)
    v_cos = torch.cos(vertical_rotation)
    v_sin = torch.sin(vertical_rotation)
    zeros = torch.zeros_like(central_rotation)
    ones = torch.ones_like(central_rotation)

    rot_z = torch.stack(
        [
            torch.stack([c_cos, -c_sin, zeros], dim=-1),
            torch.stack([c_sin, c_cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, v_cos, -v_sin], dim=-1),
            torch.stack([zeros, v_sin, v_cos], dim=-1),
        ],
        dim=-2,
    )

    # Default directions are facing in the -z direction, so origins should face opposite way
    origins = torch.stack([torch.tensor([0, 0, 1])] * size, dim=0)
    origins = (origins * radius_mean) + (origins * (torch.randn((origins.shape)) * radius_std))
    R = torch.bmm(rot_z, rot_x)  # Want to have Rx @ Ry @ origin
    t = (
        torch.bmm(R, origins.unsqueeze(-1))
        + torch.randn((size, 3, 1)) * jitter_std
        + torch.tensor(center)[None, :, None]
    )
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
class RandomCamerasDataManagerConfig(DataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: RandomCamerasDataManager)
    train_resolution: int = 64
    """Training resolution"""
    eval_resolution: int = 64
    """Evaluation resolution"""
    num_eval_angles: int = 256
    """Number of evaluation angles"""
    train_images_per_batch: int = 1
    """Number of images per batch for training"""
    eval_images_per_batch: int = 1
    """Number of images per batch for evaluation"""
    radius_mean: float = 2.5
    """Mean radius of camera orbit"""
    radius_std: float = 0.1
    """Std of radius of camera orbit"""
    focal_range: Tuple[float, float] = (0.7, 1.35)
    """Range of focal length"""
    vertical_rotation_range: Tuple[float, float] = (-90, 0)
    """Range of vertical rotation"""
    jitter_std: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    center: Tuple[float, float, float] = (0, 0, 0)
    """Center coordinate of the camera sphere"""
    horizontal_rotation_warmup: int = 0
    """How many steps until the full horizontal rotation range is used"""


class RandomCamerasDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RandomCamerasDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: RandomCamerasDataManagerConfig,
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

        if self.config.data is not None:
            CONSOLE.print("[red] --data should not be used with the RandomCamerasDataManager[/red]")
            sys.exit(1)

        cameras, _, _ = random_train_pose(
            self.config.num_eval_angles,
            self.config.eval_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            central_rotation_range=(-180, 180),
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
        )

        self.train_dataset = TrivialDataset(cameras)
        self.eval_dataset = TrivialDataset(cameras)

        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

        # pylint: disable=non-parent-init-called
        DataManager.__init__(self)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        self.train_count += 1
        horizontal_range = min((step / max(1, self.config.horizontal_rotation_warmup)), 1) * 180

        cameras, vertical_rotation, central_rotation = random_train_pose(
            self.config.train_images_per_batch,
            self.config.train_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
            central_rotation_range=(-horizontal_range, horizontal_range),
        )
        ray_bundle = cameras.generate_rays(torch.tensor(list(range(self.config.train_images_per_batch)))).flatten()

        return ray_bundle, {
            "vertical": vertical_rotation,
            "central": central_rotation,
            "initialization": True,
        }

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1

        cameras, vertical_rotation, central_rotation = random_train_pose(
            self.config.eval_images_per_batch,
            self.config.eval_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
        )
        ray_bundle = cameras.generate_rays(
            torch.tensor([[i] for i in range(self.config.train_images_per_batch)])
        ).flatten()

        return ray_bundle, {"vertical": vertical_rotation, "central": central_rotation}

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_resolution**2

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_resolution**2

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups
