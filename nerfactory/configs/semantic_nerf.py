from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base import (
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
    to_dict,
)
from nerfactory.configs.nerfw import AABBColliderConfig, FriendsDataloaderConfig
from nerfactory.configs.vanilla_nerf import VanillaNerfConfig
from nerfactory.utils.misc import DotDict


@dataclass
class SemanticNerfModelConfig(ModelConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import semantic_nerf

    _target: ClassVar[Type] = semantic_nerf.SemanticNerfModel
    collider_config: InstantiateConfig = AABBColliderConfig()
    loss_coefficients: DotDict = to_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "semantic_loss_fine": 0.05})
    num_coarse_samples: int = 64
    num_importance_samples: int = 64


@dataclass
class SemanticNerfPipelineConfig(PipelineConfig):
    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = FriendsDataloaderConfig()
    model: ModelConfig = SemanticNerfModelConfig()


@dataclass
class SemanticNerfConfig(VanillaNerfConfig):
    experiment_name: str = "friends_TBBT-big_living_room"
    method_name: str = "semantic_nerf"
    pipeline: PipelineConfig = SemanticNerfPipelineConfig()
