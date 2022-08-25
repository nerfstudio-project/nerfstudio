# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""Mipnerf 360 Configs"""
from dataclasses import dataclass
from typing import ClassVar, Dict, Type

from nerfactory.configs.base import (
    Config,
    DataloaderConfig,
    MipNerf360DataloaderConfig,
    ModelConfig,
    PipelineConfig,
    TrainerConfig,
)
from nerfactory.configs.utils import to_immutable_dict

# pylint: disable=import-outside-toplevel


@dataclass
class MipNerf360ModelConfig(ModelConfig):
    """Mipnerf 360 model config"""

    from nerfactory.models import mipnerf_360

    _target: ClassVar[Type] = mipnerf_360.MipNerf360Model
    collider_config: Dict[str, float] = to_immutable_dict({"near_plane": 0.5, "far_plane": 20.0})
    loss_coefficients: Dict[str, float] = to_immutable_dict({"ray_loss_coarse": 1.0, "ray_loss_fine": 1.0})
    num_coarse_samples: int = 128
    num_importance_samples: int = 128


@dataclass
class MipNerf360PipelineConfig(PipelineConfig):
    """Mipnerf 360 pipeline config"""

    dataloader: DataloaderConfig = MipNerf360DataloaderConfig()
    model: ModelConfig = MipNerf360ModelConfig()


@dataclass
class MipNerf360Config(Config):
    """Mipnerf 360 base config"""

    experiment_name: str = "mipnerf_360"
    method_name: str = "mipnerf_360"
    trainer: TrainerConfig = TrainerConfig(steps_per_test=200)
    pipeline: PipelineConfig = MipNerf360PipelineConfig()
