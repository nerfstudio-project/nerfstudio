from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base import (
    ColliderConfig,
    Config,
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
    TrainerConfig,
    to_dict,
)
from nerfactory.utils.misc import DotDict


@dataclass
class MipNerf360DatasetConfig(InstantiateConfig):
    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Mipnerf360
    data_directory: str = "data/mipnerf_360/garden"


@dataclass
class MipNerf360DataloaderConfig(DataloaderConfig):
    """Configuration for train/eval datasets"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader
    train_dataset: InstantiateConfig = MipNerf360DatasetConfig()


@dataclass
class MipNerf360ModelConfig(ModelConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import mipnerf_360

    _target: ClassVar[Type] = mipnerf_360.MipNerf360Model
    collider_config: InstantiateConfig = ColliderConfig(near_plane=0.5, far_plane=20.0)
    loss_coefficients: DotDict = to_dict({"ray_loss_coarse": 1.0, "ray_loss_fine": 1.0})
    num_coarse_samples: int = 128
    num_importance_samples: int = 128


@dataclass
class MipNerf360PipelineConfig(PipelineConfig):
    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = MipNerf360DataloaderConfig()
    model: ModelConfig = MipNerf360ModelConfig()


@dataclass
class MipNerf360Config(Config):
    experiment_name: str = "mipnerf_360"
    method_name: str = "vanilla_nerf"
    trainer: TrainerConfig = TrainerConfig(steps_per_test=200)
    pipeline: PipelineConfig = MipNerf360PipelineConfig()
