"""
Default test to make sure train runs
"""
import os

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from scripts.run_train import main

BLACKLIST = ["graph_nerfw.yaml", "graph_semantic_nerf.yaml", "graph_mipnerf_360.yaml"]


def set_reduced_config(config: DictConfig):
    """Reducing the config settings to speedup test"""
    config.machine.num_gpus = 0
    config.trainer.max_num_iterations = 2
    # reduce dataset factors; set dataset to test
    config.data.dataset_inputs_train.data_directory = "tests/data/lego_test"
    config.data.dataset_inputs_train.downscale_factor = 16
    config.data.dataloader_train.image_sampler.num_images_to_sample_from = 1
    config.data.dataloader_train.pixel_sampler.num_rays_per_batch = 4
    config.data.dataset_inputs_eval.data_directory = "tests/data/lego_test"
    config.data.dataset_inputs_eval.downscale_factor = 16
    # reduce graph factors
    config.graph.num_coarse_samples = 4
    config.graph.num_importance_samples = 4
    # set logging to tmp
    config.logging.writer.TensorboardWriter.log_dir = "/tmp/"
    # remove viewer
    config.viewer.enable = False

    # graph specific config settings
    if config.method_name == "instant_ngp":
        config.graph.field_implementation = "torch"

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
        main(config)

        GlobalHydra.instance().clear()


if __name__ == "__main__":
    test_run_train()
