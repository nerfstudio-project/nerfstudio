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

"""NerfW Configs"""
from dataclasses import dataclass
from typing import ClassVar, Dict, Type

from nerfactory.configs.base import (
    Config,
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
)
from nerfactory.configs.utils import to_immutable_dict

# pylint: disable=import-outside-toplevel


@dataclass
class FriendsDatasetConfig(InstantiateConfig):
    """Friends dataset config"""

    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Friends
    data_directory: str = "data/friends/TBBT-big_living_room"


@dataclass
class FriendsDataloaderConfig(DataloaderConfig):
    """Friends dataloader config"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader
    train_dataset: InstantiateConfig = FriendsDatasetConfig()
    image_dataset_type: str = "panoptic"


@dataclass
class AABBColliderConfig(InstantiateConfig):
    """AABB collider config"""

    from nerfactory.models.modules import scene_colliders

    _target: ClassVar[Type] = scene_colliders.AABBBoxCollider


@dataclass
class NerfWModelConfig(ModelConfig):
    """NerfW model config"""

    from nerfactory.models import nerfw

    _target: ClassVar[Type] = nerfw.NerfWModel
    collider_config: InstantiateConfig = AABBColliderConfig()
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "uncertainty_loss": 1.0, "density_loss": 0.01}
    )
    num_coarse_samples: int = 64
    num_importance_samples: int = 64
    uncertainty_min: float = 0.03


@dataclass
class NerfWPipelineConfig(PipelineConfig):
    """NerfW pipeline config"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = FriendsDataloaderConfig()
    model: ModelConfig = NerfWModelConfig()


@dataclass
class NerfWConfig(Config):
    """NerfW base config"""

    experiment_name: str = "friends_TBBT-big_living_room"
    method_name: str = "nerfw"
    pipeline: PipelineConfig = NerfWPipelineConfig()
