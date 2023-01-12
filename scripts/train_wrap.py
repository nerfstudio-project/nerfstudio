from __future__ import annotations
import os

os.environ["TERM"] = "dumb"
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion, method_configs
from nerfstudio.nerfstudio.data.dataparsers.blender_dataparser import (
    BlenderDataParserConfig,
)
from nerfstudio.nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.nerfstudio.configs.base_config import (
    LoggingConfig,
    ViewerConfig,
    LocalWriterConfig,
)
import scripts.train as train
import tyro
from typing import Literal, Optional
from pathlib import Path
from scripts.my_utils import *
import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from typing import TypedDict
import argparse
from datetime import datetime


from pydantic import ValidationError, Field
import pydantic.dataclasses
import dataclasses

OUTPUT_DIRECTORY = "outputs"


@dataclasses.dataclass()
class InnerArgs:
    pass
    # models: list[Literal["nerfacto", "tensorf"]]
    # datasets: list[str]


@dataclasses.dataclass()
class InnerArgsOpt:
    seg_masks: Optional[list[str]] = None


"""SUPER IMPORTANT: ALL DATACLASS' ATTRIBUTES MUST HAVE TYPE ASSIGNED, OTHERWISE THEY ARE GETTING IGNORED"""


@dataclasses.dataclass()
class MyTrainerConf(InnerArgsOpt, TrainerConfig, InnerArgs):
    logging: LoggingConfig = LoggingConfig(
        steps_per_log=200,
        max_buffer_size=200,
        local_writer=LocalWriterConfig(enable=True, max_log_size=0),
    )
    viewer: ViewerConfig = ViewerConfig(
        quit_on_train_completion=True, num_rays_per_chunk=1 << 15
    )
    save_only_latest_checkpoint: bool = False
    max_num_iterations: int = 30000
    vis: str = "viewer"


@dataclasses.dataclass()
class NerfactoConfig(MyTrainerConf):
    method_name: str = "nerfacto"
    steps_per_eval_batch: int = 500
    mixed_precision: bool = True
    pipeline: VanillaPipelineConfig = VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    )
    optimizers: dict = Field(
        default_factory=lambda: {
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
    )


@dataclasses.dataclass()
class TensorfConfig(MyTrainerConf):
    method_name: str = "tensorf"
    mixed_precision: bool = False
    pipeline: VanillaPipelineConfig = VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    )
    optimizers: dict = Field(
        default_factory=lambda: {
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=0.001),
                "scheduler": SchedulerConfig(lr_final=0.0001, max_steps=30000),
            },
            "encodings": {
                "optimizer": AdamOptimizerConfig(lr=0.02),
                "scheduler": SchedulerConfig(lr_final=0.002, max_steps=30000),
            },
        }
    )


model_confs = {"nerfacto": NerfactoConfig, "tensorf": TensorfConfig}


def dynamically_override_config(config):
    timestamp = get_timestamp()
    data_name = Path(config.data).name
    experiment_name = get_experiment_name(timestamp=timestamp)
    config.experiment_name = f"{experiment_name}-{data_name}-{config.method_name}"
    config.relative_model_dir = "."  # don't save to nerfstudio_models
    return config


def entrypoint():

    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument("--models", nargs="+", required=True)
    pre_parser.add_argument("--datasets", nargs="+", required=True)
    pre_conf, nerfstudio_args = pre_parser.parse_known_args()

    for model in pre_conf.models:
        for dataset in pre_conf.datasets:
            nerfstudio_args.insert(0, model)
            nerfstudio_args.extend(["--data", dataset])
            config = tyro.cli(AnnotatedBaseConfigUnion, args=nerfstudio_args)
            config = dynamically_override_config(config)

            out_dir = Path(config.output_dir, config.experiment_name)  # type: ignore
            out_dir.mkdir(parents=True, exist_ok=True)
            stdout_to_file(Path(out_dir, "log.txt"))
            train.main(config)


if __name__ == "__main__":
    tyro.extras.set_accent_color(None)
    entrypoint()
