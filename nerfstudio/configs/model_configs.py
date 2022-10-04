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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import dcargs

from nerfstudio.configs.base_config import Config, TrainerConfig, ViewerConfig
from nerfstudio.data.datamanagers import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.models.base_model import VanillaModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.mipnerf_360 import MipNerf360Model
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nerfw import NerfWModelConfig
from nerfstudio.models.semantic_nerf import SemanticNerfModel
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig

model_configs: Dict[str, Config] = {}
descriptions = {
    "nerfacto": "[bold green]Recommended[/bold green] Real-time model tuned for real captures. "
    + "This model will be continually updated.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for bounded synthetic data.",
    "mipnerf-360": "High quality model for unbounded 360 degree scenes. [red]*slow*",
    "mipnerf": "High quality model for bounded scenes. [red]*slow*",
    "nerfw": "Model designed to handle inconsistent appearance between images. [red]*slow*",
    "semantic-nerf": "Model that predicts dense semantic segmentations. [red]*slow*",
    "vanilla-nerf": "Original NeRF model. [red]*slow*",
    "tensorf": "Fast model designed for bounded scenes.",
}

model_configs["nerfacto"] = Config(
    method_name="nerfacto",
    trainer=TrainerConfig(steps_per_eval_batch=500, steps_per_save=2000, mixed_precision=True),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=4096, eval_num_rays_per_batch=8192
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 14),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 14),
    vis="viewer",
)

model_configs["instant-ngp"] = Config(
    method_name="instant-ngp",
    trainer=TrainerConfig(steps_per_eval_batch=500, steps_per_save=2000, mixed_precision=True),
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)

model_configs["mipnerf-360"] = Config(
    method_name="mipnerf-360",
    trainer=TrainerConfig(steps_per_eval_batch=200),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=MipNerf360DataParserConfig(), train_num_rays_per_batch=8192),
        model=VanillaModelConfig(
            _target=MipNerf360Model,
            collider_params={"near_plane": 0.5, "far_plane": 20.0},
            loss_coefficients={"ray_loss_coarse": 1.0, "ray_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=8192,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

model_configs["mipnerf"] = Config(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=8192),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=8192,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

model_configs["nerfw"] = Config(
    method_name="nerfw",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=FriendsDataParserConfig(),
        ),
        model=NerfWModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)


model_configs["semantic-nerf"] = Config(
    method_name="semantic-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=FriendsDataParserConfig(),
        ),
        model=VanillaModelConfig(
            _target=SemanticNerfModel,
            loss_coefficients={"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "semantic_loss_fine": 0.05},
            num_coarse_samples=64,
            num_importance_samples=64,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

model_configs["vanilla-nerf"] = Config(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

model_configs["tensorf"] = Config(
    method_name="tensorf",
    trainer=TrainerConfig(mixed_precision=True),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=0.001),
            "scheduler": SchedulerConfig(lr_final=0.00005, max_steps=15000),
        },
        "position_encoding": {
            "optimizer": RAdamOptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.005, max_steps=15000),
        },
        "direction_encoding": {
            "optimizer": RAdamOptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.005, max_steps=15000),
        },
    },
)


AnnotatedBaseConfigUnion = dcargs.extras.subcommand_type_from_defaults(
    defaults=model_configs, descriptions=descriptions
)
"""Union[] type over config types, annotated with default instances for use with
dcargs.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
