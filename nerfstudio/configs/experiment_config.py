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

"""Config used for running an experiment"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.configs.base_config import (
    InstantiateConfig,
    LoggingConfig,
    MachineConfig,
    ViewerConfig,
)
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

CONSOLE = Console(width=120)


@dataclass
class ExperimentConfig(InstantiateConfig):
    """Full config contents for running an experiment. Any experiment types (like training) will be
    subclassed from this, and must have their _target field defined accordingly."""

    output_dir: Path = Path("outputs")
    """relative or absolute output directory to save all checkpoints and logging"""
    method_name: Optional[str] = None
    """Method name. Required to set in python or via cli"""
    experiment_name: Optional[str] = None
    """Experiment name. If None, will automatically be set to dataset name"""
    timestamp: str = "{timestamp}"
    """Experiment timestamp."""
    machine: MachineConfig = MachineConfig()
    """Machine configuration"""
    logging: LoggingConfig = LoggingConfig()
    """Logging configuration"""
    viewer: ViewerConfig = ViewerConfig()
    """Viewer configuration"""
    pipeline: VanillaPipelineConfig = VanillaPipelineConfig()
    """Pipeline configuration"""
    optimizers: Dict[str, Any] = to_immutable_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    """Dictionary of optimizer groups and their schedulers"""
    vis: Literal["viewer", "wandb", "tensorboard"] = "wandb"
    """Which visualizer to use."""
    data: Optional[Path] = None
    """Alias for --pipeline.datamanager.dataparser.data"""
    relative_model_dir: Path = Path("nerfstudio_models/")
    """Relative path to save all checkpoints."""

    def is_viewer_enabled(self) -> bool:
        """Checks if a viewer is enabled."""
        return "viewer" == self.vis

    def is_wandb_enabled(self) -> bool:
        """Checks if wandb is enabled."""
        return "wandb" == self.vis

    def is_tensorboard_enabled(self) -> bool:
        """Checks if tensorboard is enabled."""
        return "tensorboard" == self.vis

    def set_timestamp(self) -> None:
        """Dynamically set the experiment timestamp"""
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self) -> None:
        """Dynamically set the experiment name"""
        if self.experiment_name is None:
            self.experiment_name = str(self.pipeline.datamanager.dataparser.data).replace("../", "").replace("/", "-")

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}")

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.relative_model_dir)

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")
