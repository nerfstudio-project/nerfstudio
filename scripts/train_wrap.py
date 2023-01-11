from nerfstudio.configs.method_configs import method_configs
from nerfstudio.nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.nerfstudio.configs.base_config import LoggingConfig
import train
import tyro
import dataclasses
from typing import Literal, Optional
from pathlib import Path
from my_utils import *

OUTPUT_DIRECTORY = "outputs"


@dataclasses.dataclass
class InnerArgs:
    models: list[Literal["nerfacto", "tensorf"]]
    datasets: Optional[list[str]] = None


@dataclasses.dataclass
class MyTrainerConfig(TrainerConfig):
    save_only_latest_checkpoint: bool = False
    pass


@dataclasses.dataclass
class Args(TrainerConfig, InnerArgs):
    seg_masks: Optional[list[str]] = None
    logging: LoggingConfig = LoggingConfig(steps_per_log=200, max_buffer_size=200)
    pass


def dynamically_override_config(config):
    timestamp = get_timestamp()
    dir_name = Path(config.data).name
    experiment_name = get_experiment_name(timestamp=timestamp)
    config.experiment_name = f"{experiment_name}-{dir_name}"
    config.relative_model_dir = "."  # don't save to nerfstudio_models
    return config


def entrypoint():

    _pre_conf = tyro.cli(Args)

    for model in _pre_conf.models:
        default_model_config = method_configs[model]
        config = tyro.cli(Args, default=default_model_config)
        dynamically_override_config(config)
        train.main(config)


if __name__ == "__main__":
    entrypoint()
