# pylint: disable=protected-access
"""
Default test to make sure train runs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nerfactory.configs.base import Config
from nerfactory.configs.base_configs import base_configs
from nerfactory.datamanagers.dataparsers.blender_parser import BlenderDataParserConfig
from nerfactory.engine.trainer import train_loop

BLACKLIST = ["base", "semantic-nerf", "mipnerf-360", "instant-ngp", "compound", "proposal"]


def set_reduced_config(config: Config):
    """Reducing the config settings to speedup test"""
    config.machine.num_gpus = 0
    config.trainer.max_num_iterations = 2
    # reduce dataset factors; set dataset to test
    config.pipeline.datamanager.dataparser = BlenderDataParserConfig(data_directory=Path("tests/data/lego_test"))
    config.pipeline.datamanager.train_num_images_to_sample_from = 1
    config.pipeline.datamanager.train_num_rays_per_batch = 4

    # use tensorboard logging instead of wandb
    config.logging.event_writer = "tb"
    config.logging.relative_log_dir = Path("/tmp/")

    # reduce model factors
    if hasattr(config.pipeline.model, "num_coarse_samples"):
        config.pipeline.model.num_coarse_samples = 4
    if hasattr(config.pipeline.model, "num_importance_samples"):
        config.pipeline.model.num_importance_samples = 4
    # remove viewer
    config.viewer.enable = False

    # model specific config settings
    if config.method_name == "instant-ngp":
        config.pipeline.model.field_implementation = "torch"

    return config


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_train():
    """test run train script works properly"""
    all_config_names = base_configs.keys()
    for config_name in all_config_names:
        if config_name in BLACKLIST:
            print("skipping", config_name)
            continue
        print(f"testing run for: {config_name}")
        config = base_configs[config_name]
        config = set_reduced_config(config)

        train_loop(local_rank=0, world_size=0, config=config)


if __name__ == "__main__":
    test_run_train()
