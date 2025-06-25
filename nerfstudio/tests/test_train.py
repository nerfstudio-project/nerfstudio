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
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.scripts.train import train_loop

BLACKLIST = [
    "base",
    "semantic-nerfw",
    "instant-ngp",
    "instant-ngp-bounded",
    "nerfacto-big",
    "phototourism",
    "depth-nerfacto",
    "neus",
    "generfacto",
    "neus-facto",
    "splatfacto",
    "splatfacto-big",
    "splatfacto-mcmc",
]


def set_reduced_config(config: TrainerConfig, tmp_path: Path):
    """Reducing the config settings to speedup test"""
    config.machine.device_type = "cpu"
    if hasattr(config.pipeline.model, "implementation"):
        setattr(config.pipeline.model, "implementation", "torch")
    config.mixed_precision = False
    config.use_grad_scaler = False
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
        assert isinstance(config.pipeline.model, VanillaModelConfig)
        config.pipeline.model.num_coarse_samples = 4
    if hasattr(config.pipeline.model, "num_importance_samples"):
        assert isinstance(config.pipeline.model, VanillaModelConfig)
        config.pipeline.model.num_importance_samples = 4
    # remove viewer
    config.viewer.quit_on_train_completion = True

    # timestamp & output directory
    config.set_timestamp()
    config.output_dir = tmp_path / "outputs"

    return config


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_train(tmp_path: Path):
    """test run train script works properly"""
    all_config_names = method_configs.keys()
    for config_name in all_config_names:
        if config_name in BLACKLIST:
            print("skipping", config_name)
            continue
        print(f"testing run for: {config_name}")
        config = method_configs[config_name]
        config = set_reduced_config(config, tmp_path)

        train_loop(local_rank=0, world_size=0, config=config)


def test_simple_io(tmp_path: Path):
    """test to check minimal data IO works correctly"""
    config = method_configs["vanilla-nerf"]
    config.pipeline.datamanager.dataparser = MinimalDataParserConfig(data=Path("tests/data/minimal_parser"))
    config = set_reduced_config(config, tmp_path)
    train_loop(local_rank=0, world_size=0, config=config)


if __name__ == "__main__":
    test_train()
    test_simple_io()
