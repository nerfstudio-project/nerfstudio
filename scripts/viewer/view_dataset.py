#!/usr/bin/env python
"""
view_dataset.py
"""

import logging
from datetime import timedelta

import dcargs
import torch
import yaml
from rich.console import Console

from nerfactory.configs import base as cfg
from nerfactory.configs.base_configs import AnnotatedBaseConfigUnion
from nerfactory.viewer.server import viewer_utils

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.DEBUG)

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def main(config: cfg.Config) -> None:
    """Main function."""
    if not config.viewer.enable:
        config.viewer.enable = True
        logging.info("Enabling viewer to view dataset")
    visualizer_state = viewer_utils.VisualizerState(config.viewer)
    datamanager = config.pipeline.datamanager.setup()
    visualizer_state.init_scene(dataset=datamanager.train_input_dataset, start_train=False)


if __name__ == "__main__":
    console = Console(width=120)

    instantiated_config = dcargs.cli(AnnotatedBaseConfigUnion)
    if instantiated_config.trainer.load_config:
        logging.info(f"Loading pre-set config from: {instantiated_config.trainer.load_config}")
        instantiated_config = yaml.load(instantiated_config.trainer.load_config.read_text(), Loader=yaml.Loader)

    console.rule("Config")
    console.print(instantiated_config)
    console.rule("")
    main(config=instantiated_config)
