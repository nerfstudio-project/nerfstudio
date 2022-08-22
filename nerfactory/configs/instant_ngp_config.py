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

"""Instant NGP Configs"""
from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base_config import (
    Config,
    DataloaderConfig,
    ModelConfig,
    OptimizerConfig,
    PipelineConfig,
    TrainerConfig,
    to_dict,
)
from nerfactory.configs.vanilla_nerf import BlenderDataloaderConfig
from nerfactory.utils.misc import DotDict

# pylint: disable=import-outside-toplevel


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    from nerfactory.models import instant_ngp

    _target: ClassVar[Type] = instant_ngp.NGPModel
    enable_density_field: bool = True
    enable_collider: bool = False
    field_implementation: str = "tcnn"  # torch, tcnn, ...
    loss_coefficients: DotDict = to_dict({"rgb_loss": 1.0})


@dataclass
class InstantNGPPipelineConfig(PipelineConfig):
    """Instnat NGP pipeline config"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = BlenderDataloaderConfig(train_num_rays_per_batch=8192, eval_num_rays_per_chunk=8192)
    model: ModelConfig = InstantNGPModelConfig()


@dataclass
class InstantNGPConfig(Config):
    """Instant NGP base config"""

    trainer: TrainerConfig = TrainerConfig(mixed_precision=True)
    method_name: str = "instant_ngp"
    pipeline: PipelineConfig = InstantNGPPipelineConfig()
    optimizers: DotDict = to_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": None,
            }
        }
    )
    # viewer = ViewerConfig(enable=True, num_rays_per_chunk=16384)
