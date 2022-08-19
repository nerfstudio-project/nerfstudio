from ast import Dict
from dataclasses import dataclass
from typing import Any, ClassVar, Type

from nerfactory.configs.base import (
    Config,
    DataloaderConfig,
    InstantiateConfig,
    ModelConfig,
    OptimizerConfig,
    PipelineConfig,
    TrainerConfig,
    to_dict,
)
from nerfactory.configs.vanilla_nerf import BlenderDataloaderConfig
from nerfactory.utils.misc import DotDict


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import instant_ngp

    _target: ClassVar[Type] = instant_ngp.NGPModel
    enable_density_field: bool = True
    enable_collider: bool = False
    field_implementation: str = "tcnn"  # torch, tcnn, ...
    loss_coefficients: DotDict = to_dict({"rgb_loss": 1.0})


@dataclass
class InstantNGPPipelineConfig(PipelineConfig):
    """Configuration for pipeline instantiation"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = BlenderDataloaderConfig(train_num_rays_per_batch=8192, eval_num_rays_per_chunk=8192)
    model: ModelConfig = InstantNGPModelConfig()


@dataclass
class InstantNGPConfig(Config):
    trainer: TrainerConfig = TrainerConfig(mixed_precision=True)
    method_name: str = "instant_ngp"
    pipeline: PipelineConfig = InstantNGPPipelineConfig()
    optimizers: DotDict = to_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": None,
            }
        }
    )
    # viewer = ViewerConfig(enable=True, num_rays_per_chunk=16384)
