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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type

import torch

from nerfactory.configs.utils import to_immutable_dict
from nerfactory.dataloaders.base import Dataloader, VanillaDataloader
from nerfactory.dataloaders.datasets import Blender, Dataset, Friends, Mipnerf360
from nerfactory.models.base import Model
from nerfactory.models.instant_ngp import NGPModel
from nerfactory.models.nerfw import NerfWModel
from nerfactory.optimizers.schedulers import ExponentialDecaySchedule
from nerfactory.pipelines.base import Pipeline
from nerfactory.utils import writer


# Pretty printing class
class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


# Machine related configs
@dataclass
class MachineConfig(PrintableConfig):
    """Configuration of machine setup"""

    seed: int = 42
    num_gpus: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    dist_url: str = "auto"


# Logging related configs
@dataclass
class TensorboardWriterConfig(InstantiateConfig):
    """Tensorboard Writer config"""

    _target: Type = writer.TensorboardWriter
    enable: bool = False
    relative_log_dir: Path = Path("./")
    log_dir: Optional[Path] = None  # full log dir path to be dynamically set


@dataclass
class WandbWriterConfig(InstantiateConfig):
    """WandDB Writer config"""

    _target: Type = writer.WandbWriter
    enable: bool = False
    relative_log_dir: Path = Path("./")
    log_dir: Optional[Path] = None  # full log dir path to be dynamically set


@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    enable: bool = False
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_LOAD_TIME,
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
    )
    max_log_size: int = 10
    relative_log_dir: Path = Path("./")
    log_dir: Optional[Path] = None  # full log dir path to be dynamically set


@dataclass
class LoggingConfig(PrintableConfig):
    """Configuration of loggers and profilers"""

    steps_per_log: int = 10
    max_buffer_size: int = 20
    writer: Tuple[Any, ...] = (
        TensorboardWriterConfig(enable=True),
        WandbWriterConfig(enable=False),
        LocalWriterConfig(enable=True),
    )
    # profiler logs run times of functions and prints at end of training
    enable_profiler: bool = True


# Trainer related configs
@dataclass
class TrainerConfig(PrintableConfig):
    """Configuration for training regimen"""

    steps_per_save: int = 1000
    steps_per_test: int = 500
    max_num_iterations: int = 1000000
    mixed_precision: bool = False
    relative_model_dir: Path = Path("nerfactory_models/")
    model_dir: Optional[Path] = None  # full model dir path to be dynamically set
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    load_step: Optional[int] = None
    load_config: Optional[Path] = None


# Dataset related configs
@dataclass
class DatasetConfig(InstantiateConfig):
    """Basic dataset config"""

    _target: Type = Dataset


@dataclass
class DataloaderConfig(InstantiateConfig):
    """Configuration for dataloader instantiation"""

    _target: Type = Dataloader
    image_dataset_type: str = "rgb"
    train_dataset: InstantiateConfig = DatasetConfig()
    train_num_rays_per_batch: int = 1024
    train_num_images_to_sample_from: int = -1
    eval_dataset: Optional[InstantiateConfig] = None
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    eval_num_rays_per_chunk: int = 4096


@dataclass
class BlenderDatasetConfig(InstantiateConfig):
    """Blender dataset config"""

    _target: Type = Blender
    data_directory: str = "data/blender/lego"
    scale_factor: float = 1.0
    alpha_color: str = "white"
    downscale_factor: int = 1


@dataclass
class BlenderDataloaderConfig(DataloaderConfig):
    """Blender dataloader config"""

    _target: Type = VanillaDataloader
    train_dataset: InstantiateConfig = BlenderDatasetConfig()


@dataclass
class FriendsDatasetConfig(InstantiateConfig):
    """Friends dataset config"""

    _target: Type = Friends
    data_directory: str = "data/friends/TBBT-big_living_room"


@dataclass
class FriendsDataloaderConfig(DataloaderConfig):
    """Friends dataloader config"""

    _target: Type = VanillaDataloader
    train_dataset: InstantiateConfig = FriendsDatasetConfig()
    image_dataset_type: str = "panoptic"


@dataclass
class MipNerf360DatasetConfig(InstantiateConfig):
    """Mipnerf 360 dataset config"""

    _target: Type = Mipnerf360
    data_directory: str = "data/mipnerf_360/garden"


@dataclass
class MipNerf360DataloaderConfig(DataloaderConfig):
    """Mipnerf 360 dataloader config"""

    _target: Type = VanillaDataloader
    train_dataset: InstantiateConfig = MipNerf360DatasetConfig()


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = Model
    enable_collider: bool = True
    collider_params: Dict[str, float] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    num_coarse_samples: int = 64
    num_importance_samples: int = 128
    field_implementation: Literal["torch", "tcnn"] = "torch"
    enable_density_field: bool = False
    density_field_params: Dict[str, Any] = to_immutable_dict(
        {
            "center": 0.0,  # simply set it as the center of the scene bbox
            "base_scale": 3.0,  # simply set it as the scale of the scene bbox
            "num_cascades": 1,  # if using more than 1 cascade, the `base_scale` can be smaller than scene scale.
            "resolution": 128,
            "update_every_num_iters": 16,
        }
    )


@dataclass
class InstantNGPModelConfig(ModelConfig):
    """Instant NGP Model Config"""

    _target: Type = NGPModel
    enable_density_field: bool = True
    enable_collider: bool = False
    field_implementation: Literal["torch", "tcnn"] = "tcnn"  # torch, tcnn, ...
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    num_samples: int = 1024  # instead of course/fine samples


@dataclass
class NerfWModelConfig(ModelConfig):
    """NerfW model config"""

    _target: Type = NerfWModel
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "uncertainty_loss": 1.0, "density_loss": 0.01}
    )
    num_coarse_samples: int = 64
    num_importance_samples: int = 64
    uncertainty_min: float = 0.03


# Pipeline related configs
@dataclass
class PipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = Pipeline
    dataloader: DataloaderConfig = DataloaderConfig()
    model: ModelConfig = ModelConfig()


# Viewer related configs
@dataclass
class ViewerConfig(PrintableConfig):
    """Configuration for viewer instantiation"""

    enable: bool = False
    zmq_url: str = "tcp://127.0.0.1:6000"
    min_render_image_height: int = 64
    max_render_image_height: int = 1024
    num_rays_per_chunk: int = 4096


# Optimizer related configs
@dataclass
class OptimizerConfig(InstantiateConfig):
    """Basic optimizer config with RAdam"""

    _target: Type = torch.optim.RAdam
    lr: float = 0.0005
    eps: float = 1e-08

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(params, lr=self.lr, eps=self.eps)


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = ExponentialDecaySchedule
    lr_final: float = 0.000005
    max_steps: int = 1000000

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(optimizer, lr_init, self.lr_final, self.max_steps)


@dataclass
class Config(PrintableConfig):
    """Full config contents"""

    experiment_name: str = "blender_lego"
    method_name: str = "base_method"
    base_dir: Optional[Path] = None  # base dir path to be dynamically set
    machine: MachineConfig = MachineConfig()
    logging: LoggingConfig = LoggingConfig()
    trainer: TrainerConfig = TrainerConfig()
    pipeline: PipelineConfig = PipelineConfig()
    optimizers: Dict[str, Any] = to_immutable_dict(
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
        dt_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.base_dir = Path(f"outputs/{self.experiment_name}/{self.method_name}/{dt_str}")
        self.trainer.model_dir = self.base_dir / self.trainer.relative_model_dir
        for curr_writer in self.logging.writer:
            curr_writer.log_dir = self.base_dir / curr_writer.relative_log_dir
