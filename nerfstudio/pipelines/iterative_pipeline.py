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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type, cast
from pathlib import Path, PurePath


import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
import torch.distributed as dist
import mediapy


from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.generative.stable_diffusion_img2img import StableDiffusionImg2Img

from nerfstudio.data.datamanagers.iterative_datamanager import IterativeDataManager, IterativeDataManagerConfig
from nerfstudio.models.base_model import Model, ModelConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler

import tracemalloc
tracemalloc.start()

@dataclass
class IterativePipelineConfig(VanillaPipelineConfig):
    """Iterative Pipeline Config"""

    _target: Type = field(default_factory=lambda: IterativePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = IterativeDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    generative: bool = True
    """specifies whether the pipeline is for generative models"""
    temp_save_path: Path=PurePath("")
    """save path for generated dataset"""
    prompt: str="a pineapple"

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
        self.datamanager: IterativeDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.train_dataset = None
        sd_version = "1-5"
        self.prompt = config.prompt
        self.sd = StableDiffusionImg2Img(self.device, version=sd_version)


    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
    
    @torch.no_grad()
    def save_new_train_images(self, step: int):
        print("saving training images")
        num_images_per_train = 5
        resolution = 512
        ray_bundles, cameras = self.datamanager.random_train_views(step, resolution=resolution, num_views=num_images_per_train)
        train_img_paths = []
        num_rays_per_chunk = 512

        self.model.eval()

        for i in range(num_images_per_train):
            total_rgb = []
            for j in range(0, resolution * resolution, num_rays_per_chunk):
                start_idx = (i * resolution * resolution) + j
                end_idx = (i * resolution * resolution) + j + num_rays_per_chunk

                ray_bundle = ray_bundles.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.model(ray_bundle)
                total_rgb.append(outputs['rgb'])

            img = torch.cat(total_rgb).reshape(resolution, resolution, 3).detach().cpu().numpy() # something
            save_path = PurePath(self.temp_save_path, f"img_{i}.png")
            train_img_paths.append(str(save_path))
            mediapy.write_image(str(save_path), img)

        self.train_dataset = self.datamanager.create_dataset(train_img_paths, cameras)

    @torch.no_grad()
    def update_train_images(self):
        print("updating training images")
        for i in range(len(self.train_dataset)):
        # for i in range(1):
            image = self.train_dataset.get_image(i)[None, :, :, :].permute(0, 3, 1, 2).to(self.sd.device)
            print(i)
            image = self.sd.update_img(self.prompt, "", image, num_inference_steps=50, guidance_scale=7.5)
            mediapy.write_image(self.train_dataset.image_filenames[i], image)
            del image
    
    @profiler.time_function
    def get_train_loss_dict(self, step: int):

        # create train dataset
        # for every X iterations, apply SD to all train images
        # use a data loader that allows for multiview supervision each train step
        # rerender train dataset every Y iterations

        if step == 0:
            self.save_new_train_images(step)

        # snapshot1 = tracemalloc.take_snapshot()

        if step % 20 == 0:
            self.update_train_images()

        # snapshot2 = tracemalloc.take_snapshot()
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # print("[ Top 10 differences ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        raise Exception

        return