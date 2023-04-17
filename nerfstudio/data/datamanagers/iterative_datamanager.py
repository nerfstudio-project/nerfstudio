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

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union, Optional
from pathlib import Path
from copy import deepcopy
from pathlib import Path, PurePath

import torch
from rich.progress import Console
from torch.nn import Parameter
from typing_extensions import Literal
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader, CacheDataloader

from nerfstudio.data.pixel_samplers import PixelSampler

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
):
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
class IterativeDataManagerConfig(DataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(default_factory=lambda: IterativeDataManager)
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new"""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
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
    focal_range: Tuple[float, float] = (0.6, 1.2)
    """Range of focal length"""
    vertical_rotation_range: Tuple[float, float] = (-90, 0)
    """Range of vertical rotation"""
    jitter_std: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    center: Tuple[float, float, float] = (0, 0, 0)
    """Center coordinate of the camera sphere"""
    horizontal_rotation_warmup: int = 0
    """How many steps until the full horizontal rotation range is used"""


class IterativeDataManager(DataManager):  # pylint: disable=abstract-method

    config: IterativeDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: IterativeDataManagerConfig,
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
            CONSOLE.print("[red] --data should not be used with the DreamFusionDataManager[/red]")
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

        self.train_pixel_sampler = PixelSampler(self.config.train_num_rays_per_batch)

        # pylint: disable=non-parent-init-called
        DataManager.__init__(self)

    def create_train_dataset(self, filepaths: List[Path], cameras: Cameras, camera_angles):
        dataparser_outputs = DataparserOutputs(
            image_filenames=filepaths,
            cameras=cameras,
            scene_box=SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]])),
            mask_filenames=None,
            dataparser_transform=None,
            dataparser_scale=None,
            metadata=None,
        )
        self.train_dataset = InputDataset(dataparser_outputs)
        self.camera_angles = camera_angles
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

    def random_train_views(self, step: int, resolution:int, num_views: int) -> InputDataset:
        """eawoijfaw"""
        cameras, vertical_rotation, central_rotation = random_train_pose(
            size=num_views,
            resolution=resolution,
            device= self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            central_rotation_range=(0, 360),
            vertical_rotation_range=self.config.vertical_rotation_range,
            jitter_std=self.config.jitter_std,
            center=self.config.center,
        )

        ray_bundle = cameras.generate_rays(
            torch.tensor([[i] for i in range(num_views)])
        ) 

        return ray_bundle, cameras, list(zip(vertical_rotation, central_rotation))

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

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

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_resolution**2

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_resolution**2
    
    def get_datapath(self) -> Optional[Path]:  # pylint:disable=no-self-use
        """Returns the path to the data. This is used to determine where to save camera paths."""
        if self.temp_save_path is None:
            raise Exception("need to set a temp save path")
        return Path(self.temp_save_path)

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups
