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
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast
from pathlib import Path, PurePath


import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
import torch.distributed as dist
import mediapy


from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.iterative_datamanager import IterativeDataManager, IterativeDataManagerConfig
from nerfstudio.models.base_model import Model, ModelConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler



@dataclass
class IterativePipelineConfig(VanillaPipelineConfig):
    """Iterative Pipeline Config"""

    _target: Type = field(default_factory=lambda: IterativePipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = IterativeDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    generative: bool = False
    """specifies whether the pipeline is for generative models"""
    temp_save_path: Path= PurePath("")

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
        super().__init__()
        self.temp_save_path = config.temp_save_path
        self.datamanager: IterativeDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.train_img_cameras = []

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError
    
    def save_new_train_images(self):
        ray_bundle, cameras = self.datamanager.render_train_dataset()
        model_outputs = self.model(ray_bundle)
        imgs = model_outputs.reshape() # something

        train_img_paths = []
        for i in range(10):
            img = imgs[i]
            save_path = PurePath(self.temp_save_path, f"img_{i}")
            train_img_paths.append(save_path)
            mediapy.write_image(str(save_path), img)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):

        return