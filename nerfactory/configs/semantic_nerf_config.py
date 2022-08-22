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

"""Semantic Nerf Configs"""
from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base_config import (
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
    to_dict,
)
from nerfactory.configs.nerfw_config import AABBColliderConfig, FriendsDataloaderConfig
from nerfactory.configs.vanilla_nerf import VanillaNerfConfig
from nerfactory.utils.misc import DotDict

# pylint: disable=import-outside-toplevel


@dataclass
class SemanticNerfModelConfig(ModelConfig):
    """Semantic nerf model config"""

    from nerfactory.models import semantic_nerf

    _target: ClassVar[Type] = semantic_nerf.SemanticNerfModel
    collider_config: InstantiateConfig = AABBColliderConfig()
    loss_coefficients: DotDict = to_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "semantic_loss_fine": 0.05})
    num_coarse_samples: int = 64
    num_importance_samples: int = 64


@dataclass
class SemanticNerfPipelineConfig(PipelineConfig):
    """Semantic nerf pipeline config"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = FriendsDataloaderConfig()
    model: ModelConfig = SemanticNerfModelConfig()


@dataclass
class SemanticNerfConfig(VanillaNerfConfig):
    """Semantic nerf base config"""

    experiment_name: str = "friends_TBBT-big_living_room"
    method_name: str = "semantic_nerf"
    pipeline: PipelineConfig = SemanticNerfPipelineConfig()
