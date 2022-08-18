from dataclasses import dataclass
from typing import ClassVar, Type

from nerfactory.configs.base import ModelConfig, OptimizerConfig, to_dict
from nerfactory.configs.vanilla_nerf import BlenderDataloaderConfig


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import instant_ngp

    _target = instant_ngp.NGPModel
    enable_density_field = True
    enable_collider = False
    field_implementation = "tcnn"  # torch, tcnn, ...
    loss_coefficients = to_dict({"rgb_loss": 1.0})


@dataclass
class InstantNGPPipelineConfig(PipelineConfig):
    """Configuration for pipeline instantiation"""

    from nerfactory.pipelines import base

    _target = base.Pipeline
    dataloader = BlenderDataloaderConfig(train_num_rays_per_batch=8192, eval_num_rays_per_chunk=8192)
    model = InstantNGPModelConfig()


@dataclass
class InstantNGPPConfig(Config):
    method_name = "instant_ngp"
    pipeline = InstantNGPPipelineConfig()
    optimizers = to_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(lr=3e-3, eps=1e-15),
                "scheduler": None,
            }
        }
    )
