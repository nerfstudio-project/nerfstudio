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

import dcargs
import torch
from rich import console

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
from nerfactory.models.compound import CompoundModel
from nerfactory.models.nerfw import NerfWModel
from nerfactory.models.tensorf import TensoRFModel
from nerfactory.optimizers.schedulers import ExponentialDecaySchedule
from nerfactory.pipelines.base import Pipeline
from nerfactory.utils import writer

CONSOLE = console.Console()

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
    """random seed initilization"""
    num_gpus: int = 1
    """total number of gpus available for train/eval"""
    num_machines: int = 1
    """total number of distributed machines available (for DDP)"""
    machine_rank: int = 0
    """current machine's rank (for DDP)"""
    dist_url: str = "auto"
    """distributed connection point (for DDP)"""


# Logging related configs
@dataclass
class TensorboardWriterConfig(InstantiateConfig):
    """Tensorboard Writer config"""

    _target: Type = writer.TensorboardWriter
    """target class to instantiate"""
    enable: bool = False
    """if True enables tensorboard logging, else disables"""
    relative_log_dir: Path = Path("./")
    """relative path to save all tensorboard events"""
    log_dir: dcargs.conf.Fixed[Path] = dcargs.MISSING
    """auto-populated absolute path to saved tensorboard events"""


@dataclass
class WandbWriterConfig(InstantiateConfig):
    """WandDB Writer config"""

    _target: Type = writer.WandbWriter
    """target class to instantiate"""
    enable: bool = False
    """if True enables wandb logging, else disables"""
    relative_log_dir: Path = Path("./")
    """relative path to save all wandb events"""
    log_dir: dcargs.conf.Fixed[Path] = dcargs.MISSING
    """auto-populated absolute path to saved wandb events"""


@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    """target class to instantiate"""
    enable: bool = False
    """if True enables local logging, else disables"""
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.TRAIN_RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
        writer.EventName.VIS_RAYS_PER_SEC,
        writer.EventName.TEST_RAYS_PER_SEC,
    )
    """specifies which stats will be logged/printed to terminal"""
    max_log_size: int = 10
    """maximum number of rows to print before wrapping. if 0, will print everything."""
    relative_log_dir: Path = Path("./")
    """relative local path to save all events"""
    log_dir: dcargs.conf.Fixed[Path] = dcargs.MISSING
    """auto-populated absolute local path to saved events"""

    def setup(self, banner_messages: Optional[List[str]] = None, **kwargs) -> Any:
        """Instantiate local writer

        Args:
            banner_messages: List of strings that always print at the bottom of screen. Defaults to None.
        """
        return self._target(self, banner_messages=banner_messages, **kwargs)


@dataclass
class LoggingConfig(PrintableConfig):
    """Configuration of loggers and profilers"""

    steps_per_log: int = 10
    """number of steps between logging stats"""
    max_buffer_size: int = 20
    """maximum history size to keep for computing running averages of stats.
     e.g. if 20, averages will be computed over past 20 occurances."""
    writer: Tuple[Any, ...] = (
        TensorboardWriterConfig(enable=False),
        WandbWriterConfig(enable=True),
        LocalWriterConfig(enable=True),
    )
    """list of all supported writers. Can turn on/off writers by specifying enable."""
    enable_profiler: bool = True
    """whether to enable profiling code; prints speed of functions at the end of a program.
    profiler logs run times of functions and prints at end of training"""


# Trainer related configs
@dataclass
class TrainerConfig(PrintableConfig):
    """Configuration for training regimen"""

    steps_per_save: int = 1000
    """number of steps between saves"""
    steps_per_test: int = 500
    """number of steps between eval"""
    max_num_iterations: int = 1000000
    """maximum number of iterations to run"""
    mixed_precision: bool = False
    """whether or not to use mixed precision for training"""
    relative_model_dir: Path = Path("nerfactory_models/")
    """relative path to save all checkpoints"""
    model_dir: dcargs.conf.Fixed[Path] = dcargs.MISSING
    """auto-populated absolute path to saved checkpoints"""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """optionally specify a pre-trained model directory to load from"""
    load_step: Optional[int] = None
    """optionally specify model step to load from; if none, will find most recent model in load_dir"""
    load_config: Optional[Path] = None
    """optionally specify a pre-defined config to load from"""


# Dataset related configs
@dataclass
class DataParserConfig(InstantiateConfig):
    """Basic dataset config"""

    _target: Type = DataParser
    """_target: target class to instantiate"""


@dataclass
class NerfactoryDataParserConfig(DataParserConfig):
    """Nerfactory dataset config"""

    _target: Type = Nerfactory
    """target class to instantiate"""
    data_directory: Path = Path("data/ours/posters_v3")
    """directory specifying location of data"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: int = 1
    """How much to downscale images. Defaults to 1."""
    scene_scale: float = 4.0
    """How much to scale the scene. Defaults to 0.33"""
    orientation_method: Literal["pca", "up"] = "up"
    """The method to use for orientation. Either "pca" or "up"."""


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset config"""

    _target: Type = Blender
    """target class to instantiate"""
    data_directory: Path = Path("data/blender/lego")
    """directory specifying location of data"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class FriendsDataParserConfig(DataParserConfig):
    """Friends dataset config"""

    _target: Type = Friends
    """target class to instantiate"""
    data_directory: Path = Path("data/friends/TBBT-big_living_room")
    """directory specifying location of data"""
    include_semantics: bool = True
    """whether or not to include loading of semantics data"""
    downscale_factor: int = 8
    scene_scale: float = 4.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """


@dataclass
class MipNerf360DataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset config"""

    _target: Type = Mipnerf360
    """target class to instantiate"""
    data_directory: Path = Path("data/mipnerf_360/garden")
    """directory specifying location of data"""
    downscale_factor: int = 1
    """How much to downscale images. Defaults to 1."""
    val_skip: int = 8
    """1/val_skip images to use for validation. Defaults to 8."""
    auto_scale: bool = True
    """Scale based on pose bounds. Defaults to True."""
    aabb_scale: float = 4
    """Scene scale, Defaults to 1.0."""


@dataclass
class InstantNGPDataParserConfig(DataParserConfig):
    """Instant-NGP dataset config"""

    _target: Type = InstantNGP
    """target class to instantiate"""
    data_directory: Path = Path("data/ours/posterv2")
    """directory specifying location of data"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 0.33
    """How much to scale the scene. Defaults to 0.33"""


@dataclass
class Record3DDataParserConfig(DataParserConfig):
    """Record3D dataset config"""

    _target: Type = Record3D
    """target class to instantiate"""
    data_directory: Path = Path("data/record3d/garden")
    """Location of data"""
    val_skip: int = 8
    """1/val_skip images to use for validation. Defaults to 8."""
    aabb_scale: float = 4.0
    """Scene scale, Defaults to 4.0."""
    max_dataset_size: int = 150
    """Max number of images to train on. If the dataset has
    more, images will be sampled approximately evenly. Defaults to 150."""


AnnotatedDataParserUnion = dcargs.extras.subcommand_type_from_defaults(
    {
        "nerfactory-data": NerfactoryDataParserConfig(),
        "blender-data": BlenderDataParserConfig(),
        "friends-data": FriendsDataParserConfig(),
        "mipnerf-360-data": MipNerf360DataParserConfig(),
        "instant-ngp-data": InstantNGPDataParserConfig(),
        "record3d-data": Record3DDataParserConfig(),
    },
    prefix_names=False,
)
"""Union over possible dataparser types, annotated with metadata for dcargs. This is the
same as the vanilla union, but results in shorter subcommand names."""


@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    # Note: eval_dataparser is annotated with Fixed[] to prevent dcargs from trying to
    # convert Optional[InstantiateConfig] into subcommands for choosing between None and
    # InstantiateConfig.

    _target: Type = VanillaDataManager
    """target class to instantiate"""
    train_dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """specifies the dataparser used to unpack the data"""
    train_num_rays_per_batch: int = 1024
    """number of rays per batch to use per training iteration"""
    train_num_images_to_sample_from: int = -1
    """number of images to sample during training iteration"""
    eval_dataparser: dcargs.conf.Fixed[Optional[InstantiateConfig]] = None
    """optionally specify different dataparser to use during eval; if None, uses train_dataparser"""
    eval_num_rays_per_batch: int = 1024
    """number of rays per batch to use per eval iteration"""
    eval_num_images_to_sample_from: int = -1
    """number of images to sample during eval iteration"""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """specifies the image indices to use during eval; if None, uses all"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""


@dataclass
class FriendsDataManagerConfig(VanillaDataManagerConfig):
    """Friends data manager config"""

    _target: Type = VanillaDataManager
    """target class to instantiate"""
    train_dataparser: DataParserConfig = FriendsDataParserConfig()
    """specifies the dataparser used to unpack the data"""


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = Model
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """Loss specific weights."""
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation. Defaults to 64"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation. Defaults to 128"""
    enable_density_field: bool = False
    """Whether to create a density field to filter samples."""
    density_field_params: Dict[str, Any] = to_immutable_dict(
        {
            "center": 0.0,  # simply set it as the center of the scene bbox
            "base_scale": 3.0,  # simply set it as the scale of the scene bbox
            "num_cascades": 1,  # if using more than 1 cascade, the `base_scale` can be smaller than scene scale.
            "resolution": 128,
            "update_every_num_iters": 16,
        }
    )
    """parameters to instantiate density field with"""


@dataclass
class VanillaModelConfig(InstantiateConfig):
    """Vanilla Model Config"""


@dataclass
class CompoundModelConfig(ModelConfig):
    """Compound Model Config"""

    _target: Type = CompoundModel
    enable_density_field: bool = True
    enable_collider: bool = False
    field_implementation: Literal["torch", "tcnn"] = "tcnn"  # torch, tcnn, ...
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    num_samples: int = 1024  # instead of course/fine samples
    cone_angle: float = 0.0
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""


@dataclass
class NerfWModelConfig(ModelConfig):
    """NerfW model config"""

    _target: Type = NerfWModel
    """target class to instantiate"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0, "uncertainty_loss": 1.0, "density_loss": 0.01}
    )
    """Loss specific weights."""
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation. Defaults to 64,"""
    num_importance_samples: int = 64
    """Number of samples in fine field evaluation. Defaults to 128,"""
    uncertainty_min: float = 0.03
    """This is added to the end of the uncertainty
    rendering operation. It's called 'beta_min' in other repos.
    This avoids calling torch.log() on a zero value, which would be undefined.
    Defaults to 0.03."""
    num_images: int = 10000  # TODO: don't hardcode this
    """How many images exist in the dataset."""
    appearance_embedding_dim: int = 48
    """Dimension of appearance embedding. Defaults to 48."""
    transient_embedding_dim: int = 16
    """Dimension of transient embedding. Defaults to 16."""


@dataclass
class TensoRFModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = TensoRFModel
    """target class to instantiate"""
    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 200
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (5000, 5500, 7000)
    """specifies a list of iteration step numbers to perform upsampling"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "feature_loss": 8e-5})
    """Loss specific weights."""


# Pipeline related configs
@dataclass
class PipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = Pipeline
    """target class to instantiate"""
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


# Viewer related configs
@dataclass
class ViewerConfig(PrintableConfig):
    """Configuration for viewer instantiation"""

    relative_log_filename: str = "viewer_log_filename.txt"
    """Filename to use for the log file."""
    log_filename: dcargs.conf.Fixed[Path] = dcargs.MISSING
    """Absolute path to the log file. Set automatically."""
    enable: bool = False
    """whether to enable viewer"""
    start_train: bool = True
    """whether to immediately start training upon loading viewer
    if False, will just visualize dataset but you can toggle training in viewer"""
    zmq_url: str = "tcp://127.0.0.1:6000"
    """the zmq port to connect to for communication"""
    launch_bridge_server: bool = True
    """whether or not to launch the zmq bridge server"""
    websocket_port: int = 7007
    """the default websocket port to connect to"""
    num_rays_per_chunk: int = 32768
    """number of rays per chunk to render with visualizer"""


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
    base_dir: dcargs.conf.Fixed[Path] = dcargs.MISSING
    """Experiment base directory. Set automatically."""
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
        """Make paths more specific using the current timestamp."""
        self.set_timestamp()

        # if the viewer is enabled, disable tensorboard and wandb logging
        # TODO(ethan): this is gross for now and will require a config refactor
        if self.viewer.enable:
            string = "Disabling tensorboard and wandb logging since viewer is enabled."
            CONSOLE.print(f"[bold red]{string}")
            self.logging.writer[0].enable = False
            self.logging.writer[1].enable = False
            # also disable eval steps
            string = "Disabling eval iterations since viewer is enabled."
            CONSOLE.print(f"[bold red]{string}")
            self.trainer.steps_per_test = self.trainer.max_num_iterations

    def set_timestamp(self, timestamp: Optional[str] = None) -> None:
        """Make paths in our config more specific using a timestamp.

        Args:
            timestamp: Timestamp to use, as a string. If None, defaults to the current
                time in the form YYYY-MM-DD_HHMMMSS.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.base_dir = Path(f"outputs/{self.experiment_name}/{self.method_name}/{timestamp}")
        self.trainer.model_dir = self.base_dir / self.trainer.relative_model_dir
        for curr_writer in self.logging.writer:
            curr_writer.log_dir = self.base_dir / curr_writer.relative_log_dir
        self.viewer.log_filename = self.base_dir / self.viewer.relative_log_filename
