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

"""
Put all the method implementations in one location.
"""

from typing import Dict

from nerfactory.configs.base import (
    BlenderDataloaderConfig,
    Config,
    FriendsDataloaderConfig,
    InstantNGPModelConfig,
    MipNerf360DataloaderConfig,
    ModelConfig,
    NerfWModelConfig,
    OptimizerConfig,
    PipelineConfig,
    TrainerConfig,
)
from nerfactory.models.mipnerf import MipNerfModel
from nerfactory.models.mipnerf_360 import MipNerf360Model
from nerfactory.models.semantic_nerf import SemanticNerfModel
from nerfactory.models.vanilla_nerf import NeRFModel

base_configs: Dict[str, Config] = {}
base_configs["instant_ngp"] = Config(
    method_name="instant_ngp",
    trainer=TrainerConfig(mixed_precision=True),
    pipeline=PipelineConfig(
        dataloader=BlenderDataloaderConfig(train_num_rays_per_batch=8192, eval_num_rays_per_chunk=8192),
        model=InstantNGPModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
            "scheduler": None,
        }
    },
)

base_configs["mipnerf_360"] = Config(
    experiment_name="mipnerf_360",
    method_name="mipnerf_360",
    trainer=TrainerConfig(steps_per_test=200),
    pipeline=PipelineConfig(
        dataloader=MipNerf360DataloaderConfig(),
        model=ModelConfig(
            _target=MipNerf360Model,
            collider_params={"near_plane": 0.5, "far_plane": 20.0},
            loss_coefficients={"ray_loss_coarse": 1.0, "ray_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
        ),
    ),
)

base_configs["mipnerf"] = Config(
    method_name="mipnerf",
    pipeline=PipelineConfig(
        dataloader=BlenderDataloaderConfig(),
        model=ModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
        ),
    ),
)

base_configs["nerfw"] = Config(
    experiment_name="friends_TBBT-big_living_room",
    method_name="nerfw",
    pipeline=PipelineConfig(dataloader=FriendsDataloaderConfig(), model=NerfWModelConfig()),
)

base_configs["semantic_nerf"] = Config(
    experiment_name="friends_TBBT-big_living_room",
    method_name="semantic_nerf",
    pipeline=PipelineConfig(
        dataloader=FriendsDataloaderConfig(),
        model=ModelConfig(
            _target=SemanticNerfModel,
            loss_coefficients={"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "semantic_loss_fine": 0.05},
            num_coarse_samples=64,
            num_importance_samples=64,
        ),
    ),
)

base_configs["vanilla_nerf"] = Config(
    method_name="vanilla_nerf",
    pipeline=PipelineConfig(dataloader=BlenderDataloaderConfig(), model=ModelConfig(_target=NeRFModel)),
)
