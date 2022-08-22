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
from typing import ClassVar, Type

from nerfactory.configs.base_config import (
    ColliderConfig,
    Config,
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
    TrainerConfig,
    to_dict,
)
from nerfactory.utils.misc import DotDict

# pylint: disable=import-outside-toplevel


@dataclass
class MipNerf360DatasetConfig(InstantiateConfig):
    """Mipnerf 360 dataset config"""

    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Mipnerf360
    data_directory: str = "data/mipnerf_360/garden"


@dataclass
class MipNerf360DataloaderConfig(DataloaderConfig):
    """Mipnerf 360 dataloader config"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader
    train_dataset: InstantiateConfig = MipNerf360DatasetConfig()


@dataclass
class MipNerf360ModelConfig(ModelConfig):
    """Mipnerf 360 model config"""

    from nerfactory.models import mipnerf_360

    _target: ClassVar[Type] = mipnerf_360.MipNerf360Model
    collider_config: InstantiateConfig = ColliderConfig(near_plane=0.5, far_plane=20.0)
    loss_coefficients: DotDict = to_dict({"ray_loss_coarse": 1.0, "ray_loss_fine": 1.0})
    num_coarse_samples: int = 128
    num_importance_samples: int = 128


@dataclass
class MipNerf360PipelineConfig(PipelineConfig):
    """Mipnerf 360 pipeline config"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = MipNerf360DataloaderConfig()
    model: ModelConfig = MipNerf360ModelConfig()


@dataclass
class MipNerf360Config(Config):
    """Mipnerf 360 base config"""

    experiment_name: str = "mipnerf_360"
    method_name: str = "vanilla_nerf"
    trainer: TrainerConfig = TrainerConfig(steps_per_test=200)
    pipeline: PipelineConfig = MipNerf360PipelineConfig()
