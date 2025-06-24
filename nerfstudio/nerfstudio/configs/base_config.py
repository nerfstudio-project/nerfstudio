# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Base Configs"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Type

# model instances
from nerfstudio.utils import writer


# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


# Machine related configs
@dataclass
class MachineConfig(PrintableConfig):
    """Configuration of machine setup"""

    seed: int = 42
    """random seed initialization"""
    num_devices: int = 1
    """total number of devices (e.g., gpus) available for train/eval"""
    num_machines: int = 1
    """total number of distributed machines available (for DDP)"""
    machine_rank: int = 0
    """current machine's rank (for DDP)"""
    dist_url: str = "auto"
    """distributed connection point (for DDP)"""
    device_type: Literal["cpu", "cuda", "mps"] = "cuda"
    """device type to use for training"""


@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    """target class to instantiate"""
    enable: bool = False
    """if True enables local logging, else disables"""
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.TRAIN_RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
        writer.EventName.VIS_RAYS_PER_SEC,
        writer.EventName.TEST_RAYS_PER_SEC,
        writer.EventName.ETA,
    )
    """specifies which stats will be logged/printed to terminal"""
    max_log_size: int = 10
    """maximum number of rows to print before wrapping. if 0, will print everything."""

    def setup(self, banner_messages: Optional[List[str]] = None, **kwargs) -> Any:
        """Instantiate local writer

        Args:
            banner_messages: List of strings that always print at the bottom of screen.
        """
        return self._target(self, banner_messages=banner_messages, **kwargs)


@dataclass
class LoggingConfig(PrintableConfig):
    """Configuration of loggers and profilers"""

    relative_log_dir: Path = Path("./")
    """relative path to save all logged events"""
    steps_per_log: int = 10
    """number of steps between logging stats"""
    max_buffer_size: int = 20
    """maximum history size to keep for computing running averages of stats.
     e.g. if 20, averages will be computed over past 20 occurrences."""
    local_writer: LocalWriterConfig = field(default_factory=lambda: LocalWriterConfig(enable=True))
    """if provided, will print stats locally. if None, will disable printing"""
    profiler: Literal["none", "basic", "pytorch"] = "basic"
    """how to profile the code;
        "basic" - prints speed of all decorated functions at the end of a program.
        "pytorch" - same as basic, but it also traces few training steps.
    """


# Viewer related configs
@dataclass
class ViewerConfig(PrintableConfig):
    """Configuration for viewer instantiation"""

    relative_log_filename: str = "viewer_log_filename.txt"
    """Filename to use for the log file."""
    websocket_port: Optional[int] = None
    """The websocket port to connect to. If None, find an available port."""
    websocket_port_default: int = 7007
    """The default websocket port to connect to if websocket_port is not specified"""
    websocket_host: str = "0.0.0.0"
    """The host address to bind the websocket server to."""
    num_rays_per_chunk: int = 32768
    """number of rays per chunk to render with viewer"""
    max_num_display_images: int = 512
    """Maximum number of training images to display in the viewer, to avoid lag. This does not change which images are
    actually used in training/evaluation. If -1, display all."""
    quit_on_train_completion: bool = False
    """Whether to kill the training job when it has completed. Note this will stop rendering in the viewer."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format viewer should use; jpeg is lossy compression, while png is lossless."""
    jpeg_quality: int = 75
    """Quality tradeoff to use for jpeg compression."""
    make_share_url: bool = False
    """Viewer beta feature: print a shareable URL. This flag is ignored in the legacy version of the viewer."""
    camera_frustum_scale: float = 0.1
    """Scale for the camera frustums in the viewer."""
    default_composite_depth: bool = True
    """The default value for compositing depth. Turn off if you want to see the camera frustums without occlusions."""
