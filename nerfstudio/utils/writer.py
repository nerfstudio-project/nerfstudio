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
Generic Writer class
"""
from __future__ import annotations

import enum
from abc import abstractmethod
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

from nerfstudio.configs import base_config as cfg
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.printing import human_format

CONSOLE = Console(width=120)
to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)
EVENT_WRITERS = []
EVENT_STORAGE = []
GLOBAL_BUFFER = {}


class EventName(enum.Enum):
    """Names of possible events that can be logged via Local Writer for convenience.
    see config/logging/default_logging.yaml"""

    ITER_TRAIN_TIME = "Train Iter (time)"
    TOTAL_TRAIN_TIME = "Train Total (time)"
    ITER_VIS_TIME = "Viewer Rendering (time)"
    ETA = "ETA (time)"
    TRAIN_RAYS_PER_SEC = "Train Rays / Sec"
    TEST_RAYS_PER_SEC = "Test Rays / Sec"
    VIS_RAYS_PER_SEC = "Vis Rays / Sec"
    CURR_TEST_PSNR = "Test PSNR"


class EventType(enum.Enum):
    """Possible Event types and their associated write function"""

    IMAGE = "write_image"
    SCALAR = "write_scalar"
    DICT = "write_scalar_dict"
    CONFIG = "write_config"


@check_main_thread
def put_image(name, image: TensorType["H", "W", "C"], step: int):
    """Setter function to place images into the queue to be written out

    Args:
        image: image to write out
        step: step associated with image
    """
    if isinstance(name, EventName):
        name = name.value

    EVENT_STORAGE.append({"name": name, "write_type": EventType.IMAGE, "event": image.detach().cpu(), "step": step})


@check_main_thread
def put_scalar(name: str, scalar: Any, step: int):
    """Setter function to place scalars into the queue to be written out

    Args:
        name: name of scalar
        scalar: value
        step: step associated with scalar
    """
    if isinstance(name, EventName):
        name = name.value

    EVENT_STORAGE.append({"name": name, "write_type": EventType.SCALAR, "event": scalar, "step": step})


@check_main_thread
def put_dict(name: str, scalar_dict: Dict[str, Any], step: int):
    """Setter function to place a dictionary of scalars into the queue to be written out

    Args:
        name: name of scalar dictionary
        scalar_dict: values to write out
        step: step associated with dict
    """
    EVENT_STORAGE.append({"name": name, "write_type": EventType.DICT, "event": scalar_dict, "step": step})


@check_main_thread
def put_config(name: str, config_dict: Dict[str, Any], step: int):
    """Setter function to place a dictionary of scalars into the queue to be written out

    Args:
        name: name of scalar dictionary
        scalar_dict: values to write out
        step: step associated with dict
    """
    EVENT_STORAGE.append({"name": name, "write_type": EventType.CONFIG, "event": config_dict, "step": step})


@check_main_thread
def put_time(name: str, duration: float, step: int, avg_over_steps: bool = True, update_eta: bool = False):
    """Setter function to place a time element into the queue to be written out.
    Processes the time info according to the options.

    Args:
        name: name of time item
        duration: value
        step: step associated with value
        avg_over_steps: if True, calculate and record a running average of the times
        update_eta: if True, update the ETA. should only be set for the training iterations/s
    """
    if isinstance(name, EventName):
        name = name.value

    if avg_over_steps:
        GLOBAL_BUFFER["step"] = step
        curr_event = GLOBAL_BUFFER["events"].get(name, {"buffer": [], "avg": 0})
        curr_buffer = curr_event["buffer"]
        if len(curr_buffer) >= GLOBAL_BUFFER["max_buffer_size"]:
            curr_buffer.pop(0)
        curr_buffer.append(duration)
        curr_avg = sum(curr_buffer) / len(curr_buffer)
        put_scalar(name, curr_avg, step)
        GLOBAL_BUFFER["events"][name] = {"buffer": curr_buffer, "avg": curr_avg}
    else:
        put_scalar(name, duration, step)

    if update_eta:
        ## NOTE: eta should be called with avg train iteration time
        remain_iter = GLOBAL_BUFFER["max_iter"] - step
        remain_time = remain_iter * GLOBAL_BUFFER["events"][name]["avg"]
        put_scalar(EventName.ETA, remain_time, step)
        GLOBAL_BUFFER["events"][EventName.ETA.value] = _format_time(remain_time)


@check_main_thread
def write_out_storage():
    """Function that writes all the events in storage to all the writer locations"""
    for writer in EVENT_WRITERS:
        if isinstance(writer, LocalWriter) and len(EVENT_STORAGE) > 0:
            writer.write_stats_log(EVENT_STORAGE[0]["step"])
            continue
        for event in EVENT_STORAGE:
            write_func = getattr(writer, event["write_type"].value)
            write_func(event["name"], event["event"], event["step"])

    EVENT_STORAGE.clear()


def setup_local_writer(config: cfg.LoggingConfig, max_iter: int, banner_messages: Optional[List[str]] = None) -> None:
    """Initialization of all event writers specified in config

    Args:
        config: configuration to instantiate loggers
        max_iter: maximum number of train iterations
        banner_messages: list of messages to always display at bottom of screen
    """
    if config.local_writer.enable:
        curr_writer = config.local_writer.setup(banner_messages=banner_messages)
        EVENT_WRITERS.append(curr_writer)
    else:
        CONSOLE.log("disabled local writer")

    ## configure all the global buffer basic information
    GLOBAL_BUFFER["max_iter"] = max_iter
    GLOBAL_BUFFER["max_buffer_size"] = config.max_buffer_size
    GLOBAL_BUFFER["steps_per_log"] = config.steps_per_log
    GLOBAL_BUFFER["events"] = {}


@check_main_thread
def setup_event_writer(is_wandb_enabled: bool, is_tensorboard_enabled: bool, log_dir: Path) -> None:
    """Initialization of all event writers specified in config

    Args:
        config: configuration to instantiate loggers
        max_iter: maximum number of train iterations
        banner_messages: list of messages to always display at bottom of screen
    """
    using_event_writer = False
    if is_wandb_enabled:
        curr_writer = WandbWriter(log_dir=log_dir)
        EVENT_WRITERS.append(curr_writer)
        using_event_writer = True
    if is_tensorboard_enabled:
        curr_writer = TensorboardWriter(log_dir=log_dir)
        EVENT_WRITERS.append(curr_writer)
        using_event_writer = True
    if using_event_writer:
        string = f"logging events to: {log_dir}"
    else:
        string = "Disabled tensorboard/wandb event writers"
    CONSOLE.print(f"[bold yellow]{string}")


class Writer:
    """Writer class"""

    @abstractmethod
    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        """method to write out image

        Args:
            name: data identifier
            image: rendered image to write
            step: the time step to log
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int) -> None:
        """Required method to write a single scalar value to the logger

        Args:
            name: data identifier
            scalar: value to write out
            step: the time step to log
        """
        raise NotImplementedError

    @check_main_thread
    def write_scalar_dict(self, name: str, scalar_dict: Dict[str, Any], step: int) -> None:
        """Function that writes out all scalars from a given dictionary to the logger

        Args:
            scalar_dict: dictionary containing all scalar values with key names and quantities
            step: the time step to log
        """
        for key, scalar in scalar_dict.items():
            self.write_scalar(name + "/" + key, float(scalar), step)


class TimeWriter:
    """Timer context manager that calculates duration around wrapped functions"""

    def __init__(self, writer, name, step=None, write=True):
        self.writer = writer
        self.name = name
        self.step = step
        self.write = write

        self.start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.duration = time() - self.start
        update_step = self.step is not None
        if self.write:
            self.writer.put_time(
                name=self.name,
                duration=self.duration,
                step=self.step if update_step else GLOBAL_BUFFER["max_iter"],
                avg_over_steps=update_step,
                update_eta=self.name == EventName.ITER_TRAIN_TIME,
            )


@decorate_all([check_main_thread])
class WandbWriter(Writer):
    """WandDB Writer Class"""

    def __init__(self, log_dir: Path):
        wandb.init(project="nerfstudio-project", dir=str(log_dir), reinit=True, name=log_dir.name)

    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        image = torch.permute(image, (2, 0, 1))
        wandb.log({name: wandb.Image(image)}, step=step)

    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int) -> None:
        wandb.log({name: scalar}, step=step)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        """Function that writes out the config to wandb

        Args:
            config: config dictionary to write out
        """
        wandb.config.update(config_dict)


@decorate_all([check_main_thread])
class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, log_dir: Path):
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        image = to8b(image)
        self.tb_writer.add_image(name, image, step, dataformats="HWC")

    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int) -> None:
        self.tb_writer.add_scalar(name, scalar, step)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):  # pylint: disable=unused-argument
        """Function that writes out the config to tensorboard

        Args:
            config: config dictionary to write out
        """
        self.tb_writer.add_text("config", str(config_dict))


def _cursorup(x: int):
    """utility tool to move the cursor up on the terminal

    Args:
        x: amount of lines to move cursor upward
    """
    print(f"\r\033[{x}A", end="\x1b[1K\r")


def _format_time(seconds):
    """utility tool to format time in human readable form given seconds"""
    ms = seconds % 1
    ms = ms * 1e3
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days} d, {hours} h, {minutes} m, {seconds} s"
    if hours > 0:
        return f"{hours} h, {minutes} m, {seconds} s"
    if minutes > 0:
        return f"{minutes} m, {seconds} s"
    if seconds > 0:
        return f"{seconds} s, {ms:0.3f} ms"

    return f"{ms:0.3f} ms"


@decorate_all([check_main_thread])
class LocalWriter:
    """Local Writer Class
    TODO: migrate to prettyprint

    Args:
        config: configuration to instantiate class
        banner_messages: list of messages to always display at bottom of screen
    """

    def __init__(self, config: cfg.LocalWriterConfig, banner_messages: Optional[List[str]] = None):
        self.config = config
        self.stats_to_track = [name.value for name in config.stats_to_track]
        self.keys = set()
        self.past_mssgs = ["", ""]
        self.banner_len = 0 if banner_messages is None else len(banner_messages) + 1
        if banner_messages:
            self.past_mssgs.extend(["-" * 100])
            self.past_mssgs.extend(banner_messages)
        self.has_printed = False

    def write_stats_log(self, step: int) -> None:
        """Function to write out scalars to terminal

        Args:
            step: current train step
        """
        valid_step = step % GLOBAL_BUFFER["steps_per_log"] == 0
        if valid_step:
            if not self.has_printed and self.config.max_log_size:
                CONSOLE.log(
                    f"Printing max of {self.config.max_log_size} lines. "
                    "Set flag [yellow]--logging.local-writer.max-log-size=0[/yellow] "
                    "to disable line wrapping."
                )
            latest_map, new_key = self._consolidate_events()
            self._update_header(latest_map, new_key)
            self._print_stats(latest_map)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """Function that writes out the config to local

        Args:
            config: config dictionary to write out
        """
        # TODO: implement this

    def _consolidate_events(self):
        latest_map = {}
        new_key = False
        for event in EVENT_STORAGE:
            name = event["name"]
            if name not in self.keys:
                self.keys.add(name)
                new_key = True
            latest_map[name] = event["event"]
        return latest_map, new_key

    def _update_header(self, latest_map, new_key):
        """helper to handle the printing of the header labels

        Args:
            latest_map: the most recent dictionary of stats that have been recorded
            new_key: indicator whether or not there is a new key added to logger
        """
        full_log_cond = not self.config.max_log_size and GLOBAL_BUFFER["step"] <= GLOBAL_BUFFER["steps_per_log"]
        capped_log_cond = self.config.max_log_size and (len(self.past_mssgs) - self.banner_len <= 2 or new_key)
        if full_log_cond or capped_log_cond:
            mssg = f"{'Step (% Done)':<20}"
            for name, _ in latest_map.items():
                if name in self.stats_to_track:
                    mssg += f"{name:<20} "
            self.past_mssgs[0] = mssg
            self.past_mssgs[1] = "-" * len(mssg)
            if full_log_cond or not self.has_printed:
                print(mssg)
                print("-" * len(mssg))

    def _print_stats(self, latest_map, padding=" "):
        """helper to print out the stats in a readable format

        Args:
            latest_map: the most recent dictionary of stats that have been recorded
            padding: type of characters to print to pad open space
        """
        step = GLOBAL_BUFFER["step"]
        fraction_done = step / GLOBAL_BUFFER["max_iter"]
        curr_mssg = f"{step} ({fraction_done*100:.02f}%)"
        curr_mssg = f"{curr_mssg:<20}"
        for name, v in latest_map.items():
            if name in self.stats_to_track:
                if "(time)" in name:
                    v = _format_time(v)
                elif "Rays" in name:
                    v = human_format(v)
                else:
                    v = f"{v:0.4f}"
                curr_mssg += f"{v:<20} "

        # update the history buffer
        if self.config.max_log_size:
            if not self.has_printed:
                cursor_idx = len(self.past_mssgs) - self.banner_len
                self.has_printed = True
            else:
                cursor_idx = len(self.past_mssgs)
            if len(self.past_mssgs[2:]) - self.banner_len >= self.config.max_log_size:
                self.past_mssgs.pop(2)
            self.past_mssgs.insert(len(self.past_mssgs) - self.banner_len, curr_mssg)
            _cursorup(cursor_idx)

            for i, mssg in enumerate(self.past_mssgs):
                pad_len = len(max(self.past_mssgs, key=len))
                style = "\x1b[6;30;42m" if self.banner_len and i >= len(self.past_mssgs) - self.banner_len + 1 else ""
                print(f"{style}{mssg:{padding}<{pad_len}} \x1b[0m")
        else:
            print(curr_mssg)
