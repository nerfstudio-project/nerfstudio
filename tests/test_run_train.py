# pylint: disable=protected-access
"""
Default test to make sure train runs
"""
import os

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from nerfactory.engine.trainer import train_loop
from nerfactory.utils.config import setup_config

BLACKLIST = ["graph_semantic_nerf.yaml", "graph_mipnerf_360.yaml", "graph_instant_ngp.yaml"]


def set_reduced_config(config: DictConfig):
    """Reducing the config settings to speedup test"""
    config.machine.num_gpus = 0
    config.trainer.max_num_iterations = 2
    # reduce dataset factors; set dataset to test
    # switch to using the vanilla ImageDataset class
    config.pipeline.dataloader.image_dataset_type = "rgb"

    config.pipeline.dataloader.train_dataset = {
        "_target_": "nerfactory.dataloaders.datasets.Blender",
        "data_directory": "tests/data/lego_test",
        "downscale_factor": 16,
    }
    config.pipeline.dataloader.train_num_images_to_sample_from = 1
    config.pipeline.dataloader.train_num_rays_per_batch = 4
    with open_dict(config):
        config.pipeline.dataloader.eval_dataset = {
            "_target_": "nerfactory.dataloaders.datasets.Blender",
            "data_directory": "tests/data/lego_test",
            "downscale_factor": 16,
        }

    # reduce model factors
    config.pipeline.model.num_coarse_samples = 4
    config.pipeline.model.num_importance_samples = 4
    # set logging to tmp
    config.logging.writer.TensorboardWriter.log_dir = "/tmp/"
    # remove viewer
    config.viewer.enable = False

    # model specific config settings
    if config.method_name == "instant_ngp":
        config.pipeline.model.field_implementation = "torch"

    return config


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_run_train():
    """test run train script works properly"""
    all_configs = [f for f in os.listdir("configs/") if f.endswith(".yaml")]  # relative to run path
    for config_path in all_configs:
        print(f"testing run for: {config_path}")
        if config_path in BLACKLIST:
            continue
        initialize(version_base="1.2", config_path="../configs/")  # relative to test path
        config = compose(config_path)
        config = set_reduced_config(config)
        config = setup_config(config)

        train_loop(local_rank=0, world_size=0, config=config)

        GlobalHydra.instance().clear()


if __name__ == "__main__":
    test_run_train()
