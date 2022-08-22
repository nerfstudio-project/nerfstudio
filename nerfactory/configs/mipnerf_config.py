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

"""Mipnerf Configs"""
from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base_config import (
    Config,
    DataloaderConfig,
    ModelConfig,
    PipelineConfig,
    to_dict,
)
from nerfactory.configs.vanilla_nerf import BlenderDataloaderConfig
from nerfactory.utils.misc import DotDict

# pylint: disable=import-outside-toplevel


# Differences compared to paper
#       This repo                         mipNeRF
# density = softplus(x)          density = softplus(x-1)
# rgb = sigmoid(x)               rgb = (1 + 2e) / (1 + exp(-x)) -e


@dataclass
class MipNerfModelConfig(ModelConfig):
    """Mipnerf model config"""

    from nerfactory.models import mipnerf

    _target: ClassVar[Type] = mipnerf.MipNerfModel
    loss_coefficients: DotDict = to_dict({"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0})
    num_coarse_samples: int = 128
    num_importance_samples: int = 128


@dataclass
class MipNerfPipelineConfig(PipelineConfig):
    """Mipnerf pipeline config"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = BlenderDataloaderConfig()
    model: ModelConfig = MipNerfModelConfig()


@dataclass
class MipNerfConfig(Config):
    """Mipnerf base config"""

    method_name: str = "mipnerf"
    pipeline: PipelineConfig = MipNerfPipelineConfig()
