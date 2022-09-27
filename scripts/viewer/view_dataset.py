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
    viewer: cfg.ViewerConfig = cfg.ViewerConfig(enable=True),
    log_base_dir: Path = Path("/tmp/nerfactory_viewer_logs"),
) -> None:
    """Main function."""
    assert viewer.enable
    viewer_state = viewer_utils.ViewerState(
        viewer,
        log_filename=log_base_dir / viewer.relative_log_filename,
    )
    dataset = InputDataset(dataparser.setup().get_dataset_inputs(split="train"))
    viewer_state.init_scene(dataset=dataset, start_train=False)
    logging.info("Please refresh and load page at: %s", viewer_state.viewer_url)
    time.sleep(30)  # allowing time to refresh page


if __name__ == "__main__":
    dcargs.extras.set_accent_color("bright_yellow")
    dcargs.cli(main)
