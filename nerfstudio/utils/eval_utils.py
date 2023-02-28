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
from typing import Optional, Tuple

import torch
import yaml
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.my_utils import get_step_from_ckpt_path

CONSOLE = Console(width=120, no_color=True)


def eval_load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Path:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """

    checkpoint_path = Path(config.load_ckpt)


    if checkpoint_path is None:
        CONSOLE.rule("Error", style="red")
        CONSOLE.print(
            f"Please pass the --load-ckpt <CKPT_PATH> argument.",
            justify="center",
        )
        sys.exit(1)

    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    
    if "step" not in loaded_state or not loaded_state["step"]:
        step = get_step_from_ckpt_path(checkpoint_path)
    else:
        step = loaded_state["step"]

    pipeline.load_pipeline(loaded_state["pipeline"], step)
    CONSOLE.print(
        f":white_check_mark: Done loading checkpoint from {str(checkpoint_path)}"
    )
    return checkpoint_path


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    load_ckpt: Path | None = None,
) -> Tuple[TrainerConfig, Pipeline, Path]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    if load_ckpt is not None:
        config.load_ckpt = load_ckpt
    checkpoint_path = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path
