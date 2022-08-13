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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, DictConfig


@dataclass
class MachineConfig:
    """Configuration of machine setup"""

    seed: int = MISSING
    num_gpus: int = MISSING
    num_machines: int = MISSING
    machine_rank: int = MISSING
    dist_url: str = MISSING


@dataclass
class LoggingConfig:
    """Configuration of loggers and profilers"""

    steps_per_log: int = MISSING
    max_buffer_size: int = MISSING
    writer: Dict[str, Any] = MISSING
    enable_profiler: bool = MISSING


@dataclass
class ResumeTrainConfig:
    """Configuration for loading previous checkpoints"""

    load_dir: Optional[str] = None
    load_step: Optional[int] = None


@dataclass
class TrainerConfig:
    """Configuration for training regimen"""

    model_dir: str = MISSING
    steps_per_save: int = MISSING
    steps_per_test: int = MISSING
    max_num_iterations: int = MISSING
    mixed_precision: bool = MISSING
    resume_train: ResumeTrainConfig = MISSING


@dataclass
class DataloaderConfig:
    """Configuration for train/eval datasets"""

    _target_: str = MISSING
    image_dataset_type: Optional[str] = "rgb"
    train_dataset: Dict[str, Any] = MISSING
    train_num_rays_per_batch: int = MISSING
    train_num_images_to_sample_from: int = MISSING
    eval_dataset: Optional[Dict[str, Any]] = None
    eval_image_indices: List[int] = MISSING
    eval_num_rays_per_chunk: int = MISSING


@dataclass
class ModelConfig:
    """Configuration for graph instantiation"""

    _target_: str = MISSING
    enable_collider: Optional[bool] = True
    collider_config: Dict[str, Any] = MISSING
    num_coarse_samples: int = MISSING
    num_importance_samples: int = MISSING
    loss_coefficients: Dict[str, Any] = MISSING
    # additional optional parameters here
    field_implementation: Optional[str] = "torch"
    enable_density_field: Optional[bool] = False
    density_field_config: Dict[str, Any] = MISSING


@dataclass
class PipelineConfig:
    """Configuration for pipeline instantiation"""

    _target_: str = MISSING
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
    pipeline = PipelineConfig(**config.pipeline)
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
