#!/usr/bin/env python
"""
view_dataset.py
"""

import logging
import time
from datetime import timedelta
from pathlib import Path

import torch
import tyro

from nerfstudio.configs import model_configs as cfg
from nerfstudio.data.datamanagers import AnnotatedDataParserUnion
from nerfstudio.data.utils.datasets import InputDataset
from nerfstudio.viewer.server import viewer_utils

logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def main(
    dataparser: AnnotatedDataParserUnion,
    viewer: cfg.ViewerConfig,
    log_base_dir: Path = Path("/tmp/nerfstudio_viewer_logs"),
) -> None:
    """Main function."""
    viewer_state = viewer_utils.ViewerState(
        viewer,
        log_filename=log_base_dir / viewer.relative_log_filename,
    )
    dataset = InputDataset(dataparser.setup().get_dataset_inputs(split="train"))
    viewer_state.init_scene(dataset=dataset, start_train=False)
    logging.info("Please refresh and load page at: %s", viewer_state.viewer_url)
    time.sleep(30)  # allowing time to refresh page


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)
