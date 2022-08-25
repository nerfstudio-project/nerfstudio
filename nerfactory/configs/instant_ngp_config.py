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
from typing import Any, ClassVar, Dict, Type

from nerfactory.configs.base import (
    BlenderDataloaderConfig,
    Config,
    DataloaderConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    PipelineConfig,
    TrainerConfig,
)
from nerfactory.configs.utils import to_immutable_dict

# pylint: disable=import-outside-toplevel


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    from nerfactory.models import instant_ngp

    _target: ClassVar[Type] = instant_ngp.NGPModel
    enable_density_field: bool = True
    enable_collider: bool = False
    field_implementation: str = "tcnn"  # torch, tcnn, ...
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})


@dataclass
class InstantNGPPipelineConfig(PipelineConfig):
    """Instnat NGP pipeline config"""

    dataloader: DataloaderConfig = BlenderDataloaderConfig(train_num_rays_per_batch=8192, eval_num_rays_per_chunk=8192)
    model: ModelConfig = InstantNGPModelConfig()


@dataclass
class InstantNGPConfig(Config):
    """Instant NGP base config"""

    method_name: str = "instant_ngp"
    trainer: TrainerConfig = TrainerConfig(mixed_precision=True)
    pipeline: PipelineConfig = InstantNGPPipelineConfig()
    optimizers: Dict[str, Any] = to_immutable_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": None,
            }
        }
    )
    # viewer = ViewerConfig(enable=True, num_rays_per_chunk=16384)
