from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base import (
    Config,
    DataloaderConfig,
    ModelConfig,
    PipelineConfig,
    to_dict,
)
from nerfactory.configs.vanilla_nerf import BlenderDataloaderConfig
from nerfactory.utils.misc import DotDict

# Differences compared to paper
#       This repo                         mipNeRF
# density = softplus(x)          density = softplus(x-1)
# rgb = sigmoid(x)               rgb = (1 + 2e) / (1 + exp(-x)) -e


@dataclass
class MipNerfModelConfig(ModelConfig):
    """Configuration for graph instantiation"""
    from nerfactory.models import mipnerf

    _target: ClassVar[Type] = mipnerf.MipNerfModel
    loss_coefficients: DotDict = to_dict({"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0})
    num_coarse_samples: int = 128
    num_importance_samples: int = 128


@dataclass
class MipNerfPipelineConfig(PipelineConfig):
    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = BlenderDataloaderConfig()
    model: ModelConfig = MipNerfModelConfig()


@dataclass
class MipNerfConfig(Config):
    method_name: str = "mipnerf"
    pipeline: PipelineConfig = MipNerfPipelineConfig()
