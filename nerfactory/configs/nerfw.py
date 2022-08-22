from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base import (
    Config,
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
    to_dict,
)
from nerfactory.utils.misc import DotDict


@dataclass
class FriendsDatasetConfig(InstantiateConfig):
    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Friends
    data_directory: str = "data/friends/TBBT-big_living_room"


@dataclass
class FriendsDataloaderConfig(DataloaderConfig):
    """Configuration for train/eval datasets"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader
    train_dataset: InstantiateConfig = FriendsDatasetConfig()
    image_dataset_type: str = "panoptic"


@dataclass
class AABBColliderConfig(InstantiateConfig):
    from nerfactory.models.modules import scene_colliders

    _target: ClassVar[Type] = scene_colliders.AABBBoxCollider


@dataclass
class NerfWModelConfig(ModelConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import nerfw

    _target: ClassVar[Type] = nerfw.NerfWModel
    collider_config: InstantiateConfig = AABBColliderConfig()
    loss_coefficients: DotDict = to_dict(
        {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "uncertainty_loss": 1.0, "density_loss": 0.01}
    )
    num_coarse_samples: int = 64
    num_importance_samples: int = 64
    uncertainty_min: float = 0.03


@dataclass
class NerfWPipelineConfig(PipelineConfig):
    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = FriendsDataloaderConfig()
    model: ModelConfig = NerfWModelConfig()


@dataclass
class NerfWConfig(Config):
    experiment_name: str = "friends_TBBT-big_living_room"
    method_name: str = "nerfw"
    pipeline: PipelineConfig = NerfWPipelineConfig()
