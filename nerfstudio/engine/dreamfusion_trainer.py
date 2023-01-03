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

"""Dreamfusion trainer"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.pipelines.dreamfusion_pipeline import (
    DreamfusionPipeline,
    DreamfusionPipelineConfig,
)
from nerfstudio.utils import profiler


@dataclass
class DreamfusionTrainerConfig(TrainerConfig):
    """Configuration for trainer instantiation"""

    _target: Type = field(default_factory=lambda: DreamfusionTrainer)
    """target class to instantiate"""
    pipeline: DreamfusionPipelineConfig = DreamfusionPipelineConfig()


class DreamfusionTrainer(Trainer):
    """Dreamfusion trainer"""

    pipeline: DreamfusionPipeline

    def __init__(self, config: DreamfusionTrainerConfig, local_rank: int = 0, world_size: int = 1):
        assert isinstance(config, DreamfusionTrainerConfig)
        Trainer.__init__(self, config=config, local_rank=local_rank, world_size=world_size)

    @profiler.time_function
    def train_iteration(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        self.optimizers.zero_grad_all()
        model_outputs, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step)
        latents = model_outputs["latents"]
        grad = model_outputs["grad"]
        self.grad_scaler.scale(latents).backward(gradient=grad, retain_graph=True)
        normals_loss = loss_dict["orientation_loss"] + loss_dict["pred_normal_loss"]
        if self.pipeline.config.alphas_penalty:
            normals_loss += loss_dict["alphas_loss"]
        if self.pipeline.config.opacity_penalty:
            normals_loss += loss_dict["opacity_loss"]
        self.grad_scaler.scale(normals_loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)

        return torch.tensor(0.0), loss_dict, metrics_dict
