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

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import Console
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.data.datamanagers.dreamfusion_datamanager import (
    DreamFusionDataManager,
    DreamFusionDataManagerConfig,
    random_train_pose,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.data.scene_box import SceneBox

CONSOLE = Console(width=120)


def centralized_jitter(c2w, jitter_std):
    assert c2w.shape[0] == 1
    theta = torch.randn(1) * jitter_std
    phi = torch.randn(1) * jitter_std
    R = torch.Tensor(
        [
            [torch.cos(theta), torch.sin(theta), 0],
            [-torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    R = R @ torch.Tensor(
        [
            [1, 0, 0],
            [0, torch.cos(phi), -torch.sin(phi)],
            [0, torch.sin(phi), torch.cos(phi)],
        ]
    )

    R = R.to(c2w.device)

    jittered = copy.deepcopy(c2w)

    jittered[..., :3, :3] = R @ jittered[..., :3, :3]
    jittered[..., :3, 3] = (R @ jittered[..., :3, 3].unsqueeze(-1)).squeeze(-1)

    return jittered


class TrivialDataset(InputDataset):
    """A trivial dataset with blank images for the viewer"""

    # pylint: disable=super-init-not-called
    def __init__(self, cameras: Cameras, input_img: torch.Tensor):
        self.size = cameras.size
        self.cameras = cameras
        self.alpha_color = None
        self.scene_box = SceneBox(torch.Tensor([[-1, -1, -1], [1, 1, 1]]))
        self.mask_filenames = None
        self.metadata = to_immutable_dict({})
        self.input_img = input_img

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> Dict:
        if index == 0:
            return {"image": self.input_img, "image_idx": index}
        return {"image": torch.cat([torch.ones(128, 256, 3), torch.zeros(128, 256, 3)], dim=0), "image_idx": index}


@dataclass
class DreamFusionPlusDataManagerConfig(DreamFusionDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DreamFusionPlusDataManager)
    input_image_path: Path = Path("data/generative/chair/chair.jpg")
    """Directory specifying location of input image."""
    input_image_mask_path: Optional[Path] = None
    """Directory specifying location of input image mask, if one exists"""
    radius_mean: float = 2.2
    """Mean radius of the camera sphere"""
    use_input_transient: bool = False
    """Whether to allow the input image to mask out parts of the image that give poor rgb loss"""
    input_distance: float = 1.0
    """Scalar multiple of radius_mean dictating how close we think this image was taken to the object in question.
    Large values mean this image was taken relatively far away, small values mean this was taken from relatively
    close to the object"""
    input_image_position_jitter_std: float = 0
    """Whether or not to jitter the position of the input image each time to prevent accumulation directly
    in front of the camera. If this is nonozero, it is the standard deviation of the jitter"""
    reference_directory: Path = Path("")
    """Directory specifying location of reference images."""
    anneal_camera_angles: float = 0
    """number of steps to anneal camera angles. We start near the input image and slowly work our way outwards
    as we train with this greater than 0"""
    central_rotation_range: Tuple[float, float] = (-180, 180)
    """Range of central rotation"""


class DreamFusionPlusDataManager(DreamFusionDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.
    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: DreamFusionPlusDataManagerConfig

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: DreamFusionPlusDataManagerConfig,
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

        pil_image = Image.open(self.config.input_image_path)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert image.shape[-1] == 3, "Image must be RGB"
        image = torch.from_numpy(image).to(self.device).float() / 255.0
        self.input_image = image.to(device)
        self.input_image_mask = None
        if self.config.input_image_mask_path is not None:
            pil_mask = Image.open(self.config.input_image_mask_path)
            mask = torch.from_numpy(np.array(pil_mask))
            self.input_image_mask = torch.any(mask != 0, dim=-1)

        # Transients for the input image, allows it to mask out parts of the image that are background

        original_cameras, _, _ = random_train_pose(
            1,
            1,
            device=torch.device("cpu"),
            radius_mean=self.config.radius_mean * self.config.input_distance,
            radius_std=0,
            focal_range=[1, 1],
            vertical_rotation_range=[0, 0],
            central_rotation_range=[0, 0],
            jitter_std=0,
        )

        original_cameras = Cameras(
            camera_to_worlds=original_cameras.camera_to_worlds,
            fx=float(image.shape[1]),
            fy=float(image.shape[0]),
            cx=image.shape[1] / 2,
            cy=image.shape[0] / 2,
        )
        self.original_cameras = original_cameras.to(self.device)

        cameras, _, _ = random_train_pose(
            self.config.num_eval_angles,
            self.config.eval_resolution,
            device=self.device,
            radius_mean=self.config.radius_mean,
            radius_std=self.config.radius_std,
            focal_range=self.config.focal_range,
            vertical_rotation_range=self.config.vertical_rotation_range,
            central_rotation_range=self.config.central_rotation_range,
            jitter_std=self.config.jitter_std,
        )
        cameras = cameras.to("cpu")

        cameras = Cameras(
            camera_to_worlds=torch.cat([original_cameras.camera_to_worlds, cameras.camera_to_worlds], dim=0),
            fx=torch.cat([original_cameras.fx, cameras.fx], dim=0),
            fy=torch.cat([original_cameras.fy, cameras.fy], dim=0),
            cx=torch.cat([original_cameras.cx, cameras.cx], dim=0),
            cy=torch.cat([original_cameras.cy, cameras.cy], dim=0),
            width=torch.cat([original_cameras.width, cameras.width], dim=0),
            height=torch.cat([original_cameras.height, cameras.height], dim=0),
        )

        self.train_dataset = TrivialDataset(cameras, image)
        self.eval_dataset = TrivialDataset(cameras, image)

        self.pixel_sampler = PixelSampler(self.config.train_resolution**2)

        # pylint: disable=non-parent-init-called
        DataManager.__init__(self)
        if self.config.use_input_transient:
            self.input_mask = Parameter(torch.ones(image.shape[:2]).float(), requires_grad=True)

    def next_input(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the batch for the input image"""

        if self.input_image_mask is None:
            batch = self.pixel_sampler.sample({"image": self.input_image.unsqueeze(0), "image_idx": torch.tensor([0])})
        else:
            batch = self.pixel_sampler.sample(
                {"image": self.input_image.unsqueeze(0), "image_idx": torch.tensor([0]), "mask": self.input_image_mask}
            )

        batch["input_image"] = True

        ray_indices = batch["indices"]
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.original_cameras.get_image_coords()[y, x]
        if self.config.use_input_transient:
            input_mask_pixels = torch.sigmoid(self.input_mask[y, x])
            batch["input_mask"] = input_mask_pixels

        original_camera = self.original_cameras
        if self.config.jitter_std:
            original_camera = copy.deepcopy(original_camera)
            original_camera.camera_to_worlds = centralized_jitter(
                original_camera.camera_to_worlds, self.config.input_image_position_jitter_std * min(step, 300) / 300
            )
        ray_bundle = original_camera.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
        )
        return ray_bundle, batch

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        self.train_count += 1

        # randomly choose a random camera view or the input view here
        # 0.1 - 0.5, steps 0 - 2000
        # cutoff = min(0.5, step * 0.00025)
        # cutoff = 0.1
        # if np.random.random_sample() < cutoff:
        #     return self.next_input(step)

        # # TODO Reimplement when cameras are fully working
        # if step > 2000:
        #     cameras, _, _ = random_train_pose(
        #         self.config.train_images_per_batch, self.config.train_resolution, device=self.device
        #     )

        #     ray_bundle = cameras.generate_rays(
        #         torch.tensor(list(range(self.config.train_images_per_batch)))
        #     ).flatten()
        #     return ray_bundle, {"initialization": False}

        # TODO below
        if step < self.config.anneal_camera_angles and self.config.anneal_camera_angles > 0:
            cameras, vertical_rotation, central_rotation = random_train_pose(
                self.config.train_images_per_batch,
                self.config.train_resolution,
                device=self.device,
                radius_mean=self.config.radius_mean,
                radius_std=self.config.radius_std,
                focal_range=self.config.focal_range,
                vertical_rotation_range=torch.tensor(self.config.vertical_rotation_range)
                * step
                / (self.config.anneal_camera_angles),
                central_rotation_range=torch.tensor(self.config.central_rotation_range)
                * step
                / (self.config.anneal_camera_angles),
                jitter_std=self.config.jitter_std,
            )
        else:
            cameras, vertical_rotation, central_rotation = random_train_pose(
                self.config.train_images_per_batch,
                self.config.train_resolution,
                device=self.device,
                radius_mean=self.config.radius_mean,
                radius_std=self.config.radius_std,
                focal_range=self.config.focal_range,
                vertical_rotation_range=self.config.vertical_rotation_range,
                central_rotation_range=self.config.central_rotation_range,
                jitter_std=self.config.jitter_std,
            )
        ray_bundle = cameras.generate_rays(torch.tensor(list(range(self.config.train_images_per_batch)))).flatten()

        # camera_idx = torch.randint(0, self.eval_cameras.shape[0], [1], dtype=torch.long, device=self.device)
        # ray_bundle = self.eval_cameras.generate_rays(camera_idx).flatten()

        return ray_bundle, {"vertical": vertical_rotation, "central": central_rotation, "initialization": True, "input_image": False}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        if self.config.use_input_transient:
            param_groups = {"transients": [self.input_mask]}
        else:
            return {}

        return param_groups

    def setup_train(self):
        pass

    def setup_eval(self):
        pass