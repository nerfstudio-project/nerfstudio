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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch

from nerfactory.configs.utils import to_immutable_dict

# data instances
from nerfactory.datamanagers.base import VanillaDataManager
from nerfactory.datamanagers.dataparsers.base import DataParser
from nerfactory.datamanagers.dataparsers.blender_parser import Blender
from nerfactory.datamanagers.dataparsers.friends_parser import Friends
from nerfactory.datamanagers.dataparsers.instant_ngp_parser import InstantNGP
from nerfactory.datamanagers.dataparsers.mipnerf_parser import Mipnerf360
from nerfactory.datamanagers.dataparsers.nerfactory_parser import Nerfactory
from nerfactory.datamanagers.dataparsers.record3d_parser import Record3D

# model instances
from nerfactory.models.base import Model
from nerfactory.models.instant_ngp import NGPModel
from nerfactory.models.nerfw import NerfWModel
from nerfactory.models.tensorf import TensoRFModel
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
    """Configuration of machine setup

    Args:
        seed: random seed initilization
        num_gpus: total number of gpus available for train/eval
        num_machines: total number of distributed machines available (for DDP)
        machine_rank: current machine's rank (for DDP)
        dist_url: distributed connection point (for DDP)
    """

    seed: int = 42
    num_gpus: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    dist_url: str = "auto"


# Logging related configs
@dataclass
class TensorboardWriterConfig(InstantiateConfig):
    """Tensorboard Writer config

    Args:
        _target: target class to instantiate
        enable: if True enables tensorboard logging, else disables
        relative_log_dir: relative path to save all tensorboard events
        log_dir: auto populated absolute path to saved tensorboard events [Do not set!]
    """

    _target: Type = writer.TensorboardWriter
    enable: bool = False
    relative_log_dir: Path = Path("./")
    log_dir: Optional[Path] = None  # full log dir path to be dynamically set


@dataclass
class WandbWriterConfig(InstantiateConfig):
    """WandDB Writer config

    Args:
        _target: target class to instantiate
        enable: if True enables wandb logging, else disables
        relative_log_dir: relative path to save all wandb events
        log_dir: auto populated absolute path to saved wandb events [Do not set!]
    """

    _target: Type = writer.WandbWriter
    enable: bool = False
    relative_log_dir: Path = Path("./")
    log_dir: Optional[Path] = None  # full log dir path to be dynamically set


@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config

    Args:
        _target: target class to instantiate
        enable: if True enables local logging, else disables
        stats_to_track: specifies which stats will be logged/printed to terminal
        max_log_size: maximum number of rows to print before wrapping. if 0, will print everything.
        relative_log_dir: relative local path to save all events
        log_dir: auto populated absolute local path to saved events [Do not set!]
    """

    _target: Type = writer.LocalWriter
    enable: bool = False
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.TRAIN_RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
        writer.EventName.VIS_RAYS_PER_SEC,
        writer.EventName.TEST_RAYS_PER_SEC,
    )
    max_log_size: int = 10
    relative_log_dir: Path = Path("./")
    log_dir: Optional[Path] = None  # full log dir path to be dynamically set

    def setup(self, banner_messages: Optional[List[str]] = None, **kwargs) -> Any:
        """Instantiate local writer

        Args:
            banner_messages: List of strings that always print at the bottom of screen. Defaults to None.
        """
        return self._target(self, banner_messages=banner_messages, **kwargs)


@dataclass
class LoggingConfig(PrintableConfig):
    """Configuration of loggers and profilers

    Args:
        steps_per_log: number of steps between logging stats
        max_buffer_size: maximum history size to keep for computing running averages of stats.
            e.g. if 20, averages will be computed over past 20 occurances.
        writer: list of all supported writers. Can turn on/off writers by specifying enable.
        enable_profiler: whether to enable profiling code; prints speed of functions at the end of a program.
            profiler logs run times of functions and prints at end of training
    """

    steps_per_log: int = 10
    max_buffer_size: int = 20
    writer: Tuple[Any, ...] = (
        TensorboardWriterConfig(enable=True),
        WandbWriterConfig(enable=False),
        LocalWriterConfig(enable=True),
    )
    enable_profiler: bool = True


# Trainer related configs
@dataclass
class TrainerConfig(PrintableConfig):
    """Configuration for training regimen

    Args:
        steps_per_save: number of steps between saves
        steps_per_test: number of steps between eval
        max_num_iterations: maximum number of iterations to run
        mixed_precision: whether or not to use mixed precision for training
        relative_model_dir: relative path to save all checkpoints
        model_dir: auto populated absolute path to saved checkpoints [Do not set!]
        load_dir: optionally specify a pre-trained model directory to load from
        load_step: optionally specify model step to load from; if none, will find most recent model in load_dir
        load_config: optionally specify a pre-defined config to load from
    """

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
class DataParserConfig(InstantiateConfig):
    """Basic dataset config

    Args:
        _target: target class to instantiate
    """

    _target: Type = DataParser


@dataclass
class NerfactoryDataParserConfig(DataParserConfig):
    """Nerfactory dataset config

    Args:
        _target: target class to instantiate
        data_directory: directory specifying location of data
        scale_factor: How much to scale the camera origins by.
        downscale_factor: How much to downscale images. Defaults to 1.
        scene_scale: How much to scale the scene. Defaults to 0.33
        orientation_method: The method to use for orientation. Either "pca" or "up".
    """

    _target: Type = Nerfactory
    data_directory: Path = Path("data/ours/posterv2")
    scale_factor: float = 1.0
    downscale_factor: int = 1
    scene_scale: float = 4.0
    orientation_method: Literal["pca", "up"] = "up"


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset config

    Args:
        _target: target class to instantiate
        data_directory: directory specifying location of data
        scale_factor: How much to scale the camera origins by.
        alpha_color: alpha color of background
    """

    _target: Type = Blender
    data_directory: Path = Path("data/blender/lego")
    scale_factor: float = 1.0
    alpha_color: str = "white"


@dataclass
class FriendsDataParserConfig(DataParserConfig):
    """Friends dataset config

    Args:
        _target: target class to instantiate
        data_directory: directory specifying location of data
        include_semantics: whether or not to include loading of semantics data
    """

    _target: Type = Friends
    data_directory: Path = Path("data/friends/TBBT-big_living_room")
    include_semantics: bool = True


@dataclass
class MipNerf360DataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset config

    Args:
        _target: target class to instantiate
        data_directory: directory specifying location of data
        downscale_factor: How much to downscale images. Defaults to 1.
        val_skip: 1/val_skip images to use for validation. Defaults to 8.
        auto_scale: Scale based on pose bounds. Defaults to True.
        aabb_scale: Scene scale, Defaults to 1.0.
    """

    _target: Type = Mipnerf360
    data_directory: Path = Path("data/mipnerf_360/garden")
    downscale_factor: int = 1
    val_skip: int = 8
    auto_scale: bool = True
    aabb_scale = 4


@dataclass
class InstantNGPDataParserConfig(DataParserConfig):
    """Instant-NGP dataset config

    Args:
        data_directory: directory specifying location of data
        scale_factor: How much to scale the camera origins by.
        scene_scale: How much to scale the scene. Defaults to 0.33
    """

    _target: Type = InstantNGP
    data_directory: Path = Path("data/ours/posterv2")
    scale_factor: float = 1.0
    scene_scale: float = 0.33


@dataclass
class Record3DDataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset config

    Args:
        data_directory: Location of data
        val_skip: 1/val_skip images to use for validation. Defaults to 8.
        aabb_scale: Scene scale, Defaults to 4.0.
        max_dataset_size: Max number of images to train on. If the dataset has
            more, images will be sampled approximately evenly. Defaults to 150.
    """

    _target: Type = Record3D
    data_directory: Path = Path("data/record3d/garden")
    val_skip: int = 8
    aabb_scale = 4.0
    max_dataset_size: int = 150


@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration

    Args:
        _target: target class to instantiate
        train_dataparser: specifies the dataparser used to unpack the data
        train_num_rays_per_batch: number of rays per batch to use per training iteration
        train_num_images_to_sample_from: number of images to sample during training iteration
        eval_dataparser: optionally specify different dataparser to use during eval; if None, uses train_dataparser
        eval_image_indices: specifies the image indices to use during eval; if None, uses all
        eval_num_rays_per_chunk: specifies number of rays per chunk during eval
    """

    _target: Type = VanillaDataManager
    train_dataparser: DataParserConfig = BlenderDataParserConfig()
    train_num_rays_per_batch: int = 1024
    train_num_images_to_sample_from: int = -1
    eval_dataparser: Optional[InstantiateConfig] = None
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    eval_num_rays_per_chunk: int = 4096


@dataclass
class FriendsDataManagerConfig(VanillaDataManagerConfig):
    """Friends data manager config

    Args:
        _target: target class to instantiate
        train_dataparser: specifies the dataparser used to unpack the data
    """

    _target: Type = VanillaDataManager
    train_dataparser: DataParserConfig = FriendsDataParserConfig()


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation

    Args:
        _target: target class to instantiate
        enable_collider: Whether to create a scene collider to filter rays.
        collider_params: parameters to instantiate scene collider with
        loss_coefficients: Loss specific weights.
        num_coarse_samples: Number of samples in coarse field evaluation. Defaults to 64,
        num_importance_samples: Number of samples in fine field evaluation. Defaults to 128,
        field_implementation (str): one of "torch" or "tcnn", or other fields in 'field_implementation_to_class'
        enable_density_field: Whether to create a density field to filter samples.
        density_field_params: parameters to instantiate density field with
    """

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
    """Instant NGP Model Config

    Args:
        _target: target class to instantiate
        enable_collider: Whether to create a scene collider to filter rays.
        loss_coefficients: Loss specific weights.
        field_implementation (str): one of "torch" or "tcnn", or other fields in 'field_implementation_to_class'
        enable_density_field: Whether to create a density field to filter samples.
        num_samples: Number of samples in field evaluation. Defaults to 1024,
    """

    _target: Type = NGPModel
    enable_collider: bool = False
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    field_implementation: Literal["torch", "tcnn"] = "tcnn"  # torch, tcnn, ...
    enable_density_field: bool = True
    num_samples: int = 1024  # instead of course/fine samples


@dataclass
class NerfWModelConfig(ModelConfig):
    """NerfW model config

    Args:
        _target: target class to instantiate
        loss_coefficients: Loss specific weights.
        num_coarse_samples: Number of samples in coarse field evaluation. Defaults to 64,
        num_importance_samples: Number of samples in fine field evaluation. Defaults to 128,
        uncertainty_min (float, optional): This is added to the end of the uncertainty
                rendering operation. It's called 'beta_min' in other repos.
                This avoids calling torch.log() on a zero value, which would be undefined.
                Defaults to 0.03.
        num_images: How many images exist in the dataset.
        appearance_embedding_dim: Dimension of appearance embedding. Defaults to 48.
        transient_embedding_dim: Dimension of transient embedding. Defaults to 16.
    """

    _target: Type = NerfWModel
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "uncertainty_loss": 1.0, "density_loss": 0.01}
    )
    num_coarse_samples: int = 64
    num_importance_samples: int = 64
    uncertainty_min: float = 0.03
    num_images: int = 10000  # TODO: don't hardcode this
    appearance_embedding_dim: int = 48
    transient_embedding_dim: int = 16


@dataclass
class TensoRFModelConfig(ModelConfig):
    """TensoRF model config

    Args:
        _target: target class to instantiate
        init_resolution: initial render resolution
        final_resolution: final render resolution
        upsampling_iters: specifies a list of iteration step numbers to perform upsampling
        loss_coefficients: Loss specific weights.
    """

    _target: Type = TensoRFModel
    init_resolution: int = 128
    final_resolution: int = 200
    upsampling_iters: Tuple[int, ...] = (5000, 5500, 7000)
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "feature_loss": 8e-5})


# Pipeline related configs
@dataclass
class PipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation

    Args:
        _target: target class to instantiate
        datamanager: specifies the datamanager config
        model: specifies the model config
    """

    _target: Type = Pipeline
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    model: ModelConfig = ModelConfig()


# Viewer related configs
@dataclass
class ViewerConfig(PrintableConfig):
    """Configuration for viewer instantiation

    Args:
        log_filename: Filename to use for the log file. Defaults to None. If None, no log file is created.
        enable: whether to enable viewer
        start_train: whether to immediately start training upon loading viewer
            if False, will just visualize dataset but you can toggle training in viewer
        zmq_url: the zmq port to connect to for communication
        launch_bridge_server: whether or not to launch the zmq bridge server
        websocket_port: the default websocket port to connect to
        num_rays_per_chunk: number of rays per chunk to render with visualizer
    """

    log_filename: Optional[Path] = None
    enable: bool = False
    start_train: bool = True
    zmq_url: str = "tcp://127.0.0.1:6000"
    launch_bridge_server: bool = True
    websocket_port: int = 7007
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
        if self.trainer.model_dir is None:
            self.trainer.model_dir = self.base_dir / self.trainer.relative_model_dir
        for curr_writer in self.logging.writer:
            curr_writer.log_dir = self.base_dir / curr_writer.relative_log_dir
        if self.viewer.log_filename is None:
            self.viewer.log_filename = self.base_dir / "viewer_log_filename.txt"
