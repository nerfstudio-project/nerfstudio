# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import yaml
from rich.console import Console

from nerfstudio.configs import base_config as cfg
from nerfstudio.pipelines.base_pipeline import Pipeline

console = Console(width=120)


def eval_setup(config_path: Path) -> Tuple[cfg.Config, Pipeline, Path]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.trainer.load_dir = config.get_checkpoint_dir()
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=True)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path = eval_load_checkpoint(config.trainer, pipeline)

    return config, pipeline, checkpoint_path
