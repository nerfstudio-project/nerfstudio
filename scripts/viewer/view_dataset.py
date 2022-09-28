#!/usr/bin/env python
"""
view_dataset.py
"""

import logging
import time
from datetime import timedelta
from pathlib import Path

import dcargs
import torch

from nerfactory.configs import base as cfg
from nerfactory.datamanagers.base import AnnotatedDataParserUnion
from nerfactory.datamanagers.datasets import InputDataset
from nerfactory.viewer.server import viewer_utils

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def main(
    dataparser: AnnotatedDataParserUnion,
    viewer: cfg.ViewerConfig,
    log_base_dir: Path = Path("/tmp/nerfactory_viewer_logs"),
) -> None:
    """Main function."""
    viewer_state = viewer_utils.ViewerState(config.viewer)
    datamanager = config.pipeline.datamanager.setup()
    viewer_state.init_scene(dataset=datamanager.train_input_dataset, start_train=False)
    logging.info("Please refresh and load page at: %s", viewer_state.viewer_url)
    time.sleep(30)  # allowing time to refresh page


if __name__ == "__main__":
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(main)
