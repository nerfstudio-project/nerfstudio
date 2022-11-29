#!/usr/bin/env python
"""
view_dataset.py
"""

from rich.console import Console

CONSOLE = Console(width=120)
import time
from datetime import timedelta
from pathlib import Path

import torch
import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers import AnnotatedDataParserUnion
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.viewer.server import viewer_utils

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def main(
    dataparser: AnnotatedDataParserUnion,
    viewer: ViewerConfig,
    log_base_dir: Path = Path("/tmp/nerfstudio_viewer_logs"),
) -> None:
    """Main function."""
    viewer_state = viewer_utils.ViewerState(
        viewer,
        log_filename=log_base_dir / viewer.relative_log_filename,
    )
    dataset = InputDataset(dataparser.setup().get_dataparser_outputs(split="train"))
    viewer_state.init_scene(dataset=dataset, start_train=False)
    CONSOLE.log("Please refresh and load page at: %s", viewer_state.viewer_url)
    time.sleep(30)  # allowing time to refresh page


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)
