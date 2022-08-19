from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base import (
    Config,
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    PipelineConfig,
)


@dataclass
class BlenderDatasetConfig(InstantiateConfig):
    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Blender
    data_directory: str = "data/blender/lego"
    scale_factor: float = 1.0
    alpha_color: str = "white"
    downscale_factor: int = 1


@dataclass
class BlenderDataloaderConfig(DataloaderConfig):
    """Configuration for train/eval datasets"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader
    train_dataset: InstantiateConfig = BlenderDatasetConfig()


@dataclass
class VanillaNerfModelConfig(ModelConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import vanilla_nerf

    _target: ClassVar[Type] = vanilla_nerf.NeRFModel


@dataclass
class VanillaNerfPipelineConfig(PipelineConfig):
    """Configuration for pipeline instantiation"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = BlenderDataloaderConfig()
    model: ModelConfig = VanillaNerfModelConfig()


@dataclass
class VanillaNerfConfig(Config):
    method_name: str = "vanilla_nerf"
    pipeline: PipelineConfig = VanillaNerfPipelineConfig()
