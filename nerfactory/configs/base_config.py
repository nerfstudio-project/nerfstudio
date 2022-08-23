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

"""Base Configs"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar

import torch

from nerfactory.utils.misc import DotDict

# pylint: disable=import-outside-toplevel

# cannot use mutable types directly within dataclass; abstracting default factory calls
def to_dict(d: Dict[str, Any]):
    """Method to convert mutable dict to default factory dict

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda: DotDict(d))


def to_list(l: List[Any]):
    """Method to convert mutable list to default factory list

    Args:
        l: list to convert into default factory list for dataclass
    """
    return field(default_factory=lambda: l)


class InstantiateConfig:  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: ClassVar[Type]

    def setup(self, **kwargs) -> Any:
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
class TensorboardWriterConfig(InstantiateConfig):
    from nerfactory.utils import writer

    _target: ClassVar[Type] = writer.TensorboardWriter
    log_dir: str = "./"


@dataclass
class WandbWriterConfig(InstantiateConfig):
    from nerfactory.utils import writer

    _target: ClassVar[Type] = writer.WandbWriter
    log_dir: str = "./"


@dataclass
class LocalWriterConfig(InstantiateConfig):
    from nerfactory.utils import writer

    _target: ClassVar[Type] = writer.LocalWriter
    stats_to_track: List[writer.EventName] = to_list(
        [
            writer.EventName.ITER_LOAD_TIME,
            writer.EventName.ITER_TRAIN_TIME,
            writer.EventName.RAYS_PER_SEC,
            writer.EventName.CURR_TEST_PSNR,
        ]
    )
    max_log_size: int = 10
    log_dir: str = "./"


@dataclass
class LoggingConfig:
    """Configuration of loggers and profilers"""

    steps_per_log: int = 10
    max_buffer_size: int = 20
    writer: List[Any] = to_list([TensorboardWriterConfig(), LocalWriterConfig()])
    # profiler logs run times of functions and prints at end of training
    enable_profiler: bool = True


@dataclass
class TrainerConfig:
    """Configuration for training regimen"""

    model_dir: str = "nerfactory_models/"
    steps_per_save: int = 1000
    steps_per_test: int = 500
    max_num_iterations: int = 1000000
    mixed_precision: bool = False
    # optional parameters if we want to resume training
    load_dir: Optional[str] = None
    load_step: Optional[int] = None


@dataclass
class DatasetConfig(InstantiateConfig):
    """Basic dataset config"""

    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Dataset


@dataclass
class DataloaderConfig(InstantiateConfig):
    """Configuration for train/eval datasets"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.Dataloader
    image_dataset_type: str = "rgb"
    train_dataset: InstantiateConfig = DatasetConfig()
    train_num_rays_per_batch: int = 1024
    train_num_images_to_sample_from: int = -1
    eval_dataset: Optional[InstantiateConfig] = None
    eval_image_indices: List[int] = to_list([0])
    eval_num_rays_per_chunk: int = 4096


@dataclass
class BlenderDatasetConfig(InstantiateConfig):
    """Blender dataset config"""

    from nerfactory.dataloaders import datasets

    _target: ClassVar[Type] = datasets.Blender
    data_directory: str = "data/blender/lego"
    scale_factor: float = 1.0
    alpha_color: str = "white"
    downscale_factor: int = 1


@dataclass
class BlenderDataloaderConfig(DataloaderConfig):
    """Blender dataloader config"""

    from nerfactory.dataloaders import base

    _target: ClassVar[Type] = base.VanillaDataloader
    train_dataset: InstantiateConfig = BlenderDatasetConfig()


@dataclass
class ColliderConfig(InstantiateConfig):
    """Basic collider config: near/far"""

    from nerfactory.models.modules import scene_colliders

    _target: ClassVar[Type] = scene_colliders.NearFarCollider
    near_plane: float = 2.0
    far_plane: float = 6.0


@dataclass
class DensityFieldConfig(InstantiateConfig):
    """Basic density field config"""

    from nerfactory.fields.density_fields import density_grid

    _target: ClassVar[Type] = density_grid.DensityGrid
    center: float = 0.0  # simply set it as the center of the scene bbox
    base_scale: float = 3.0  # simply set it as the scale of the scene bbox
    num_cascades: int = 1  # if using more than 1 cascade, the `base_scale` can be smaller than scene scale.
    resolution: int = 128
    update_every_num_iters: int = 16


@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for graph instantiation"""

    from nerfactory.models import base

    _target: ClassVar[Type] = base.Model
    enable_collider: bool = True
    collider_config: InstantiateConfig = ColliderConfig()
    loss_coefficients: DotDict = to_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    num_coarse_samples: int = 64
    num_importance_samples: int = 128
    field_implementation: str = "torch"
    enable_density_field: bool = False
    density_field_config: InstantiateConfig = DensityFieldConfig()


@dataclass
class PipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    from nerfactory.pipelines import base

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
    """Basic optimizer config with RAdam"""

    _target: ClassVar[Type] = torch.optim.RAdam
    lr: float = 0.0005
    eps: float = 1e-08

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params=None, **kwargs) -> TypeVar:
        """Returns the instantiated object using the config."""
        return self._target(params, lr=self.lr, eps=self.eps)


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    from nerfactory.optimizers import schedulers

    _target: ClassVar[Type] = schedulers.ExponentialDecaySchedule
    lr_final: float = 0.000005
    max_steps: int = 1000000

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, optimizer=None, lr_init=None, **kwargs) -> TypeVar:
        """Returns the instantiated object using the config."""
        return self._target(optimizer, lr_init, self.lr_final, self.max_steps)


@dataclass
class Config:
    """Full config contents"""

    experiment_name: str = "blender_lego"
    method_name: str = "base_method"
    machine: MachineConfig = MachineConfig()
    logging: LoggingConfig = LoggingConfig()
    trainer: TrainerConfig = TrainerConfig()
    pipeline: PipelineConfig = PipelineConfig()
    optimizers: DotDict = to_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    viewer: ViewerConfig = ViewerConfig()

    def __post_init__(self):
        """Convert logging directories to more specific filepaths"""
        dt_str: str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_dir: str = f"outputs/{self.experiment_name}/{self.method_name}/{dt_str}"
        self.trainer.model_dir = f"{base_dir}/{self.trainer.model_dir}"
        for writer in self.logging.writer:
            writer.log_dir = f"{base_dir}/{writer.log_dir}"


def setup_config(config_name: str):
    """Mapping from config name to actual config; list of all pre-implemented NeRF models"""
    if config_name == "instant_ngp":
        from nerfactory.configs.instant_ngp_config import InstantNGPConfig

        return InstantNGPConfig()
    if config_name == "mipnerf_360":
        from nerfactory.configs.mipnerf_360_config import MipNerf360Config

        return MipNerf360Config()
    if config_name == "mipnerf":
        from nerfactory.configs.mipnerf_config import MipNerfConfig

        return MipNerfConfig()
    if config_name == "nerfw":
        from nerfactory.configs.nerfw_config import NerfWConfig

        return NerfWConfig()
    if config_name == "semantic_nerf":
        from nerfactory.configs.semantic_nerf_config import SemanticNerfConfig

        return SemanticNerfConfig()
    if config_name == "vanilla_nerf":
        from nerfactory.configs.vanilla_nerf_config import VanillaNerfConfig

        return VanillaNerfConfig()

    raise NotImplementedError
