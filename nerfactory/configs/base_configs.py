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

import copy
from typing import Dict, Type

import dcargs
from typeguard import typeguard_ignore

from nerfactory.configs.base import (
    BlenderDataParserConfig,
    Config,
    LoggingConfig,
    MipNerf360DataParserConfig,
    ModelConfig,
    NerfWModelConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
    TensoRFModelConfig,
    TrainerConfig,
    VanillaDataManagerConfig,
    ViewerConfig,
)
from nerfactory.datamanagers.dataparsers.friends_parser import FriendsDataParserConfig
from nerfactory.models.compound import CompoundModelConfig
from nerfactory.models.instant_ngp import InstantNGPModelConfig
from nerfactory.models.mipnerf import MipNerfModel
from nerfactory.models.mipnerf_360 import MipNerf360Model
from nerfactory.models.semantic_nerf import SemanticNerfModel
from nerfactory.models.vanilla_nerf import NeRFModel

base_configs: Dict[str, Config] = {}
base_configs["instant-ngp"] = Config(
    method_name="instant-ngp",
    trainer=TrainerConfig(steps_per_eval_batch=500, steps_per_save=2000, mixed_precision=True),
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(train_dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(enable=True),
    logging=LoggingConfig(event_writer="none"),
)

base_configs["mipnerf-360"] = Config(
    method_name="mipnerf-360",
    trainer=TrainerConfig(steps_per_eval_batch=200),
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(
            train_dataparser=MipNerf360DataParserConfig(), train_num_rays_per_batch=8192
        ),
        model=ModelConfig(
            _target=MipNerf360Model,
            collider_params={"near_plane": 0.5, "far_plane": 20.0},
            loss_coefficients={"ray_loss_coarse": 1.0, "ray_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=8192,
        ),
    ),
)

base_configs["mipnerf"] = Config(
    method_name="mipnerf",
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(train_dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=8192),
        model=ModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=8192,
        ),
    ),
)

base_configs["nerfw"] = Config(
    method_name="nerfw",
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(
            train_dataparser=FriendsDataParserConfig(),
        ),
        model=NerfWModelConfig(),
    ),
)


base_configs["semantic-nerf"] = Config(
    method_name="semantic-nerf",
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(
            train_dataparser=FriendsDataParserConfig(),
        ),
        model=ModelConfig(
            _target=SemanticNerfModel,
            loss_coefficients={"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "semantic_loss_fine": 0.05},
            num_coarse_samples=64,
            num_importance_samples=64,
        ),
    ),
)

base_configs["vanilla-nerf"] = Config(
    method_name="vanilla-nerf",
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(
            train_dataparser=BlenderDataParserConfig(),
        ),
        model=ModelConfig(_target=NeRFModel),
    ),
)

base_configs["tensorf"] = Config(
    method_name="tensorf",
    trainer=TrainerConfig(mixed_precision=True),
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(
            train_dataparser=BlenderDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": OptimizerConfig(lr=0.001),
            "scheduler": SchedulerConfig(lr_final=0.00005, max_steps=15000),
        },
        "position_encoding": {
            "optimizer": OptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.005, max_steps=15000),
        },
        "direction_encoding": {
            "optimizer": OptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.005, max_steps=15000),
        },
    },
)

base_configs["compound"] = Config(
    method_name="compound",
    trainer=TrainerConfig(mixed_precision=True),
    pipeline=PipelineConfig(
        datamanager=VanillaDataManagerConfig(train_dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=8192),
        model=CompoundModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(enable=True),
    logging=LoggingConfig(event_writer="none"),
)


@typeguard_ignore  # TypeGuard doesn't understand the generic alias that's returned here.
def _make_base_config_subcommand_type() -> Type[Config]:
    """Generate a Union[] type over the possible base config types, with runtime
    annotations containing default values. Used to generate CLIs.

    Returns:
        An annotated Union type, which can be used to pick a base configuration.
    """
    # When a base config is used to generate a CLI: replace the auto-generated timestamp
    # with {timestamp}. This makes the CLI helptext (and, for zsh, autocomplete
    # generation) consistent everytime you run a script with --help.
    #
    # Note that when a config is instantiated with dcargs.cli(), the __post_init__
    # called when the config is instantiated will set the correct timestamp.
    base_configs_placeholder_timestamp = {}
    for name, config in base_configs.items():
        base_configs_placeholder_timestamp[name] = copy.deepcopy(config)
        base_configs_placeholder_timestamp[name].populate_dynamic_fields(timestamp="{timestamp}")

    return dcargs.extras.subcommand_type_from_defaults(base_configs_placeholder_timestamp)


AnnotatedBaseConfigUnion = _make_base_config_subcommand_type()
"""Union[] type over config types, annotated with default instances for use with
dcargs.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
