# pylint: disable=protected-access
"""
Default test to make sure train runs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.minimal_dataparser import MinimalDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from scripts.train import train_loop

BLACKLIST = [
    "base",
    "semantic-nerfw",
    "instant-ngp",
    "instant-ngp-bounded",
    "nerfacto",
    "volinga",
    "phototourism",
    "depth-nerfacto",
    "nerfplayer-ngp",
    "nerfplayer-nerfacto",
    "neus",
]


def set_reduced_config(config: TrainerConfig):
    """Reducing the config settings to speedup test"""
    config.machine.num_gpus = 0
    config.max_num_iterations = 2
    # reduce dataset factors; set dataset to test
    config.pipeline.datamanager.dataparser = BlenderDataParserConfig(data=Path("tests/data/lego_test"))
    config.pipeline.datamanager.train_num_images_to_sample_from = 1
    config.pipeline.datamanager.train_num_rays_per_batch = 4

    # use tensorboard logging instead of wandb
    config.vis = "tensorboard"
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
def test_train():
    """test run train script works properly"""
    all_config_names = method_configs.keys()
    for config_name in all_config_names:
        if config_name in BLACKLIST:
            print("skipping", config_name)
            continue
        print(f"testing run for: {config_name}")
        config = method_configs[config_name]
        config = set_reduced_config(config)

        train_loop(local_rank=0, world_size=0, config=config)


def test_simple_io():
    """test to check minimal data IO works correctly"""
    config = method_configs["vanilla-nerf"]
    config.pipeline.datamanager.dataparser = MinimalDataParserConfig(data=Path("tests/data/minimal_parser"))
    config = set_reduced_config(config)
    train_loop(local_rank=0, world_size=0, config=config)


if __name__ == "__main__":
    test_train()
    test_simple_io()
