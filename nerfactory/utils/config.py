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

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar

from omegaconf import MISSING, DictConfig
from nerfactory.dataloaders.base import VanillaDataloader
from nerfactory.models.base import Model
from ..pipelines.base import Pipeline


class InstantiateConfig:
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: ClassVar[Type]

    def setup(self) -> TypeVar:
        """Returns the instantiated object using the config."""
        return self._target(self)


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
    writer: Dict[str, Any] = MISSING
    # profiler logs run times of functions and prints at end of training
    enable_profiler: bool = True


@dataclass
class ResumeTrainConfig:
    """Configuration for loading previous checkpoints"""

    load_dir: Optional[str] = None
    load_step: Optional[int] = None


@dataclass
class TrainerConfig:
    """Configuration for training regimen"""

    model_dir: str = "./checkpoints"
    steps_per_save: int = 1000
    steps_per_test: int = 500
    max_num_iterations: int = 1000000
    mixed_precision: bool = False
    resume_train: ResumeTrainConfig = MISSING


@dataclass
class VanillaDataloaderConfig(InstantiateConfig):
    """Configuration for train/eval datasets"""

    _target: ClassVar[Type] = VanillaDataloader
    image_dataset_type: Optional[str] = "rgb"
    train_dataset: Dict[str, Any] = MISSING
    train_num_rays_per_batch: int = MISSING
    train_num_images_to_sample_from: int = MISSING
    eval_dataset: Optional[Dict[str, Any]] = None
    eval_image_indices: Optional[List[int]] = None
    eval_num_rays_per_chunk: int = MISSING


@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for graph instantiation"""

    _target: ClassVar[Type] = Model
    enable_collider: Optional[bool] = True
    collider_config: Dict[str, Any] = MISSING
    loss_coefficients: Dict[str, Any] = MISSING
    # additional optional parameters here
    field_implementation: Optional[str] = "torch"
    enable_density_field: Optional[bool] = False
    density_field_config: Dict[str, Any] = MISSING


@dataclass
class PipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: ClassVar[Type] = Pipeline
    dataloader: DataloaderConfig = MISSING
    model: ModelConfig = MISSING


@dataclass
class ViewerConfig:
    """Configuration for viewer instantiation"""

    enable: bool = MISSING
    zmq_url: str = MISSING
    min_render_image_height: int = MISSING
    max_render_image_height: int = MISSING
    num_rays_per_chunk: int = MISSING


@dataclass
class Config:
    """Full config contents"""

    machine: MachineConfig = MISSING
    logging: LoggingConfig = MISSING
    trainer: TrainerConfig = MISSING
    experiment_name: str = MISSING
    method_name: str = MISSING
    optimizers: Dict[str, Any] = MISSING
    viewer: ViewerConfig = MISSING
    pipeline: PipelineConfig = MISSING
    # additional optional parameters here
    hydra: Optional[Dict[str, Any]] = None


def setup_config(config: DictConfig) -> Config:
    """helper that creates a typed config from the DictConfig

    Args:
        config (DictConfig): configuration to convert

    Returns:
        Config: typed version of the input configuration
    """
    machine = MachineConfig(**config.machine)
    logging = LoggingConfig(**config.logging)
    trainer = TrainerConfig(**config.trainer)
    experiment_name = config.experiment_name
    method_name = config.method_name
    dataloader = DataloaderConfig(**config.pipeline.dataloader)
    model = ModelConfig(**config.pipeline.model)
    pipeline = PipelineConfig(
        _target_=config.pipeline._target_, dataloader=dataloader, model=model  # pylint:disable=protected-access
    )
    optimizers = config.optimizers
    viewer = ViewerConfig(**config.viewer)

    return Config(
        machine=machine,
        logging=logging,
        trainer=trainer,
        experiment_name=experiment_name,
        method_name=method_name,
        pipeline=pipeline,
        optimizers=optimizers,
        viewer=viewer,
    )
