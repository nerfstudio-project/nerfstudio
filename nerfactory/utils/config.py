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

"""Structured config classes"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, overload

import torch

from .misc import DotDict

# cannot use mutable types directly within dataclass; abstracting default factory calls
to_dict = lambda d: field(default_factory=lambda: DotDict(d))
to_list = lambda l: field(default_factory=lambda: l)


class InstantiateConfig:
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: ClassVar[Type]

    def setup(self, **kwargs) -> TypeVar:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


@dataclass
class MachineConfig:
    """Configuration of machine setup"""

    seed: int = 42
    num_gpus: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    dist_url: str = "auto"


@dataclass
class LoggingConfig:
    """Configuration of loggers and profilers"""

    steps_per_log: int = 10
    max_buffer_size: int = 20
    writer: Dict[str, "Any"] = to_dict(
        {
            "TensorboardWriter": {"log_dir": "./"},
            "LocalWriter": {
                "log_dir": "./",
                "stats_to_track": ["ITER_LOAD_TIME", "ITER_TRAIN_TIME", "RAYS_PER_SEC", "CURR_TEST_PSNR"],
                "max_log_size": 10,  # if 0, logs everything with no erasing
            },
        }
    )
    # profiler logs run times of functions and prints at end of training
    enable_profiler: bool = True


@dataclass
class TrainerConfig:
    """Configuration for training regimen"""

    model_dir: str = "./checkpoints"
    steps_per_save: int = 1000
    steps_per_test: int = 500
    max_num_iterations: int = 1000000
    mixed_precision: bool = False
    # optional parameteres if we want to resume training
    load_dir: Optional[str] = None
    load_step: Optional[int] = None


@dataclass
class BlenderDatasetConfig(InstantiateConfig):
    # TODO move
    from ..dataloaders import datasets

    _target: ClassVar[Type] = datasets.Blender
    data_directory: str = "data/blender/lego"
    scale_factor: float = 1.0
    alpha_color: str = "white"
    downscale_factor: int = 1


@dataclass
class DataloaderConfig(InstantiateConfig):
    """Configuration for train/eval datasets"""

    from ..dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader  # TODO: change
    image_dataset_type: str = "rgb"
    train_dataset: InstantiateConfig = BlenderDatasetConfig()  # TODO: change
    train_num_rays_per_batch: int = 1024
    train_num_images_to_sample_from: int = -1
    eval_dataset: Optional[InstantiateConfig] = None
    eval_image_indices: List[int] = to_list([0])
    eval_num_rays_per_chunk: int = 4096


@dataclass
class ColliderConfig(InstantiateConfig):
    from ..models.modules import scene_colliders

    _target: ClassVar[Type] = scene_colliders.NearFarCollider
    near_plane: float = 2.0
    far_plane: float = 6.0


@dataclass
class DensityFieldConfig(InstantiateConfig):
    from ..fields.density_fields import density_grid

    _target: ClassVar[Type] = density_grid.DensityGrid
    center: float = 0.0  # simply set it as the center of the scene bbox
    base_scale: float = 3.0  # simply set it as the scale of the scene bbox
    num_cascades: int = 1  # if using more than 1 cascade, the `base_scale` can be smaller than scene scale.
    resolution: int = 128
    update_every_num_iters: int = 16


@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for graph instantiation"""

    from ..models import vanilla_nerf  # TODO: change

    _target: ClassVar[Type] = vanilla_nerf.NeRFModel  # TODO: change
    enable_collider: Optional[bool] = True
    collider_config: InstantiateConfig = ColliderConfig()
    loss_coefficients: Dict[str, Any] = to_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    num_coarse_samples: int = 64
    num_importance_samples: int = 128
    field_implementation: str = "torch"
    enable_density_field: bool = False
    density_field_config: InstantiateConfig = DensityFieldConfig()


@dataclass
class PipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    from ..pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = DataloaderConfig()
    model: ModelConfig = ModelConfig()


@dataclass
class ViewerConfig:
    """Configuration for viewer instantiation"""

    enable: bool = False
    zmq_url: str = "tcp://127.0.0.1:6000"
    min_render_image_height: int = 64
    max_render_image_height: int = 1024
    num_rays_per_chunk: int = 4096


@dataclass
class OptimizerConfig(InstantiateConfig):
    _target: ClassVar[Type] = torch.optim.RAdam
    lr: float = 0.0005

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params) -> TypeVar:
        """Returns the instantiated object using the config."""
        return self._target(params, lr=self.lr)


@dataclass
class SchedulerConfig(InstantiateConfig):
    from ..optimizers import schedulers

    _target: ClassVar[Type] = schedulers.ExponentialDecaySchedule
    lr_final: float = 0.000005
    max_steps: int = 1000000

    def setup(self, optimizer, lr_init) -> TypeVar:
        """Returns the instantiated object using the config."""
        return self._target(optimizer, lr_init, self.lr_final, self.max_steps)


@dataclass
class Config:
    """Full config contents"""

    machine: MachineConfig = MachineConfig()
    logging: LoggingConfig = LoggingConfig()
    trainer: TrainerConfig = TrainerConfig()
    experiment_name: str = "blender_lego"
    method_name: str = "vanilla_nerf"
    # TODO: figure out a better way to do the optimizers
    optimizers: Dict[str, Any] = to_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    viewer: ViewerConfig = ViewerConfig()
    pipeline: PipelineConfig = PipelineConfig()
    # additional optional parameters here
    hydra: Optional[Dict[str, Any]] = None


def setup_config():
    return Config()
