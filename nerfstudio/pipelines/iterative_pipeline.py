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
A pipeline that saves outputs from the nerf to iteratively apply SD on .
"""

import os
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Dict, List, Mapping, Optional, Type, cast

import mediapy
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.data.datamanagers.iterative_datamanager import (
    IterativeDataManager,
    IterativeDataManagerConfig,
)
from nerfstudio.data.datamanagers.dreamfusion_datamanager import (
    DreamFusionDataManagerConfig,
)
from nerfstudio.generative.stable_diffusion_img2img import StableDiffusionImg2Img
from nerfstudio.generative.stable_diffusion_utils import PositionalTextEmbeddings
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler

tracemalloc.start()


@dataclass
class IterativePipelineConfig(VanillaPipelineConfig):
    """Iterative Pipeline Config"""

    _target: Type = field(default_factory=lambda: IterativePipeline)
    """target class to instantiate"""
    datamanager2: DataManagerConfig = DreamFusionDataManagerConfig()
    """specifies the datamanager config"""
    datamanager: DataManagerConfig = IterativeDataManagerConfig()
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    model2: ModelConfig=ModelConfig()
    generative: bool = True
    """specifies whether the pipeline is for generative models"""
    temp_save_path: str = "iterative_assets"
    """save path for generated dataset"""
    prompt: str = "a photo of a pineapple"

class IterativePipeline(VanillaPipeline):
    """Pipeline with logic for saving outputs from nerf for use as training images."""

    def __init__(
        self,
        config: IterativePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.temp_save_path = config.temp_save_path
        self.datamanager.temp_save_path = self.temp_save_path
        sd_version = "1-5"

        self._sd = StableDiffusionImg2Img(self.device, version=sd_version)
        self.text_embeddings = PositionalTextEmbeddings(
            base_prompt=config.prompt,
            top_prompt=config.prompt + ", overhead view",
            side_prompt=config.prompt + ", side view",
            back_prompt=config.prompt + ", back view",
            front_prompt=config.prompt + ", front view",
            stable_diffusion=self._sd,
            positional_prompting="discrete",
        )

        if not Path(self.temp_save_path).exists():
            Path.mkdir(Path(self.temp_save_path))
            Path.mkdir(Path(self.temp_save_path) / "rendered_images")
            Path.mkdir(Path(self.temp_save_path) / "generated_images")

    def init_first_model():
        return
    def init_second_model():
        return

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @torch.no_grad()
    def save_images_and_cameras(self, step: int):
        num_images = 3
        resolution = 64

        ray_bundles, cameras_small, vertical_rotation, central_rotation = self.datamanager.random_train_views(step, resolution=resolution, num_views=num_images)
         
        train_img_paths = []
        temp_path = "/home/terrance/nerfactory/iterative_assets/latent_nerf_images"
        self.model.eval()

        for i in range(num_images):
            cur_bundle = ray_bundles[:, :, i]
            rgb = self.model.get_outputs_for_camera_ray_bundle(cur_bundle)["rgb"]

            img = rgb.detach().cpu().numpy()  # something
            save_path = PurePath(temp_path, f"img_{i}.png")
            train_img_paths.append(str(save_path))
            mediapy.write_image(str(save_path), img)

    
        torch.save(train_img_paths, str(PurePath(temp_path, 'img_paths.pt')))
        torch.save(cameras_small, str(PurePath(temp_path, 'cameras_small.pt')))
        camera_angles = list(zip(vertical_rotation, central_rotation))
        torch.save(camera_angles, str(PurePath(temp_path, 'camera_angles.pt')))

        del ray_bundles
        cameras = self.datamanager.random_train_views(step, resolution * 8, num_images, 
                                                            vertical_rotation, central_rotation, cameras_only=True)
        torch.save(cameras, str(PurePath(temp_path, 'cameras.pt')))


    @torch.no_grad()
    def save_new_train_images(self, step: int):
        print("saving training images")
        num_images_per_train = 25
        resolution = 512
        ray_bundles, cameras, camera_angles = self.datamanager.random_train_views(
            step, resolution=resolution, num_views=num_images_per_train
        )
        train_img_paths = []

        for i in range(num_images_per_train):
            cur_bundle = ray_bundles[:, :, i]
            rgb = self.model.get_outputs_for_camera_ray_bundle(cur_bundle)["rgb"]

            img = rgb.detach().cpu().numpy()  # something
            save_path = PurePath(self.temp_save_path, "rendered_images", f"img_{i}.png")
            train_img_paths.append(str(save_path))
            mediapy.write_image(str(save_path), img)

        self.datamanager.create_train_dataset(train_img_paths, cameras, camera_angles)

    @torch.no_grad()
    def update_train_images(self, step: int):
        print("updating training images")
        generated_img_paths = []

        for i in range(len(self.datamanager.train_dataset)):
            camera_angles = self.datamanager.camera_angles[i]
            text_embedding = self.text_embeddings.get_text_embedding(camera_angles[0], camera_angles[1])

            image = self.datamanager.train_dataset.get_image(i)[None, :, :, :].permute(0, 3, 1, 2).to(self._sd.device)
            image = self._sd.update_img(text_embedding, image, step)
            save_path = PurePath(self.temp_save_path, "generated_images", f"img_{i}.png")
            generated_img_paths.append(str(save_path))
            mediapy.write_image(str(save_path), image)

        cameras = self.datamanager.train_dataset.cameras

        self.datamanager.create_train_dataset(generated_img_paths, cameras, self.datamanager.camera_angles)

    def update_single_image(self, step: int):

        to_update_idx = torch.rand(1) * len(self.datamanager.train_dataset)
        to_update_idx = int(torch.floor(to_update_idx))
        print(to_update_idx)

        # re render image
        cur_cam = self.datamanager.train_dataset.cameras[to_update_idx]
        ray_bundle = cur_cam.generate_rays(0)[:, :, None]

        # replace image with new image
        cur_bundle = ray_bundle[:, :, 0]
        rgb = self.model.get_outputs_for_camera_ray_bundle(cur_bundle)["rgb"]

        rendered = rgb.detach().cpu().numpy()  # something
        save_path = PurePath(self.temp_save_path, "rendered_images", f"img_{to_update_idx}.png")
        mediapy.write_image(str(save_path), rendered)

        # run sd on image
        camera_angles = self.datamanager.camera_angles[to_update_idx]
        text_embedding = self.text_embeddings.get_text_embedding(camera_angles[0], camera_angles[1])
        img = rgb[None, :, :, :].permute(0, 3, 1, 2).to(self._sd.device)
        img = self._sd.update_img(text_embedding, img, step)

        # recreate dataset
        save_path = PurePath(self.temp_save_path, "generated_images", f"img_{to_update_idx}.png")
        mediapy.write_image(str(save_path), img)

        img_paths = self.datamanager.train_dataset.image_filenames
        self.datamanager.create_train_dataset(
            img_paths, self.datamanager.train_dataset.cameras, self.datamanager.camera_angles
        )

    def load_images(self):
        temp_path = "/home/terrance/nerfactory/iterative_assets/latent_nerf_images"

        train_img_paths = torch.load(str(PurePath(temp_path, 'img_paths.pt')))
        cameras = torch.load(str(PurePath(temp_path, 'cameras.pt')))
        camera_angles = torch.load(str(PurePath(temp_path, 'camera_angles.pt')))

        self.datamanager.create_train_dataset(train_img_paths, cameras, camera_angles)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):

        # new plan 
        # start with sds loss / dreamfusion
        # at 6k iters? switch to this iterative process with small updates

        # print('saving images')
        # self.save_images_and_cameras(step)
        # raise Exception

        # update_steps = [0, 2000, 4000, 6000, 10000, 15000]

        # if step in (0, 2000):
        #     self.save_new_train_images(step)
        #     self.update_train_images(step)
        # elif step > 2000 and step % 250 == 0:
        #     self.update_single_image(step)

        if step == 0:
            self.load_images()

        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)

        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict