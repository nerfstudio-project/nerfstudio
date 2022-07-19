# Copyright 2022 The Plenoptix Team. All rights reserved.
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


import enum
import os
import sys
from abc import abstractmethod
from time import time
from typing import Dict

import imageio
import numpy as np
import torch
from omegaconf import ListConfig
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

import wandb
from pyrad.utils.config import LoggingConfig
from pyrad.utils.decorators import check_main_thread, decorate_all

to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)
EVENT_WRITERS = []
EVENT_STORAGE = []
GLOBAL_BUFFER = {}


class EventName(enum.Enum):
    """Names of possible events that can be logged via Local Writer for convenience.
    see config/logging/default_logging.yaml"""

    ITER_LOAD_TIME = "Data Load (time)"
    ITER_TRAIN_TIME = "Train Iter (time)"
    TOTAL_TRAIN_TIME = "Train Total (time)"
    ETA = "ETA (time)"
    RAYS_PER_SEC = "Rays Per Sec"
    CURR_TEST_PSNR = "Test PSNR"


class EventType(enum.Enum):
    """Possible Event types and their associated write function"""

    IMAGE = "write_image"
    SCALAR = "write_scalar"
    DICT = "write_scalar_dict"


@check_main_thread
def put_image(name, image: TensorType["H", "W", "C"], step: int):
    """Setter function to place images into the queue to be written out"""
    if isinstance(name, EventName):
        name = name.value

    EVENT_STORAGE.append({"name": name, "write_type": EventType.IMAGE, "event": image, "step": step})


@check_main_thread
def put_scalar(name: str, scalar: float, step: int):
    """Setter function to place scalars into the queue to be written out"""
    if isinstance(name, EventName):
        name = name.value

    EVENT_STORAGE.append({"name": name, "write_type": EventType.SCALAR, "event": scalar, "step": step})


@check_main_thread
def put_dict(name: str, scalar_dict: float, step: int):
    """Setter function to place a dictionary of scalars into the queue to be written out"""
    EVENT_STORAGE.append({"name": name, "write_type": EventType.DICT, "event": scalar_dict, "step": step})


@check_main_thread
def put_time(name: str, duration: float, step: int, avg_over_steps: bool = True, update_eta: bool = False):
    """Setter function to place a time element into the queue to be written out

    Processes the time info according to the options:
    avg_over_steps (bool): if True, calculate and record a running average of the times
    update_eta (bool): if True, update the ETA. should only be set for the training iterations/s
    """
    if isinstance(name, EventName):
        name = name.value

    if avg_over_steps:
        GLOBAL_BUFFER["step"] = step
        curr_event = GLOBAL_BUFFER["events"].get(name, {"buffer": [], "avg": 0})
        curr_buffer = curr_event["buffer"]
        curr_avg = curr_event["avg"]
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


@check_main_thread
def write_out_storage():
    """Function that writes all the events in storage to all the writer locations"""
    for writer in EVENT_WRITERS:
        for event in EVENT_STORAGE:
            write_func = getattr(writer, event["write_type"].value)
            if event["write_type"] == EventType.DICT:
                write_func(event["event"], event["step"])
                if isinstance(writer, LocalWriter):
                    continue
            else:
                write_func(event["name"], event["event"], event["step"])
                if isinstance(writer, LocalWriter):
                    break
    EVENT_STORAGE.clear()


@check_main_thread
def setup_event_writers(config: LoggingConfig, max_iter: int) -> None:
    """Initialization of all event writers specified in config"""
    for writer_type in config.writer:
        writer_class = getattr(sys.modules[__name__], writer_type)
        writer_config = config.writer[writer_type]
        if writer_type == "LocalWriter":
            curr_writer = writer_class(writer_config.log_dir, writer_config.stats_to_track, writer_config.max_log_size)
        else:
            curr_writer = writer_class(writer_config.log_dir)
        EVENT_WRITERS.append(curr_writer)

    ## configure all the global buffer basic information
    GLOBAL_BUFFER["max_iter"] = max_iter
    GLOBAL_BUFFER["max_buffer_size"] = config.max_buffer_size
    GLOBAL_BUFFER["steps_per_log"] = config.steps_per_log
    GLOBAL_BUFFER["events"] = {}


class Writer:
    """Writer class"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    @abstractmethod
    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        """_summary_

        Args:
            name (str): data identifier
            image (TensorType["H", "W", "C"]): rendered image to write
            step (int): the time step to log
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(self, name: str, scalar: float, step: int) -> None:
        """Required method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            step (int): the time step to log
        """
        raise NotImplementedError

    @check_main_thread
    def write_scalar_dict(self, scalar_dict: Dict[str, float], step: int) -> None:
        """Function that writes out all scalars from a given dictionary to the logger

        Args:
            scalar_dict (dict): dictionary containing all scalar values with key names and quantities
            step (int): the time step to log
        """
        for name, scalar in scalar_dict.items():
            self.write_scalar(name, scalar, step)


class TimeWriter:
    """Timer context manager that calculates duration around wrapped functions"""

    def __init__(self, writer, name, step=None, write=True):
        self.writer = writer
        self.name = name
        self.step = step
        self.write = write

        self.start = None
        self.duration = None

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

    def __init__(self, log_dir: str):
        super().__init__(log_dir)
        wandb.init(dir=log_dir)

    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        """_summary_

        Args:
            name (str): data identifier
            image (TensorType["H", "W", "C"]): rendered image to write
        """
        image = torch.permute(image, (2, 0, 1))
        wandb.log({name: wandb.Image(image)}, step=step)

    def write_scalar(self, name: str, scalar: float, step: int) -> None:
        """Wandb method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            scalar (float): scalar value to write
        """
        wandb.log({name: scalar}, step=step)


@decorate_all([check_main_thread])
class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, log_dir: str):
        super().__init__(log_dir)
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        """_summary_

        Args:
            name (str): data identifier
            image (TensorType["H", "W", "C"]): rendered image to write
        """
        image = to8b(image)
        self.tb_writer.add_image(name, image, step, dataformats="HWC")

    def write_scalar(self, name: str, scalar: float, step: int) -> None:
        """Tensorboard method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            scalar (float): scalar value to write
        """
        self.tb_writer.add_scalar(name, scalar, step)


def _cursorup(x: int):
    """utility tool to move the cursor up on the terminal

    Args:
        x (int): amount of lines to move cursor upward
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
class LocalWriter(Writer):
    """Local Writer Class"""

    def __init__(self, log_dir: str, stats_to_track: ListConfig, max_log_size: int = 0):
        """
        Args:
            stats_to_track (ListConfig): the names of stats that should be logged.
            max_log size (int): max number of lines that will be logged to teminal.
        """
        super().__init__(log_dir)
        self.stats_to_track = [EventName[name].value for name in stats_to_track]
        self.max_log_size = max_log_size
        self.keys = set()
        self.past_mssgs = ["", ""]

    def write_image(self, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
        if name in self.stats_to_track:
            image = to8b(image)
            image_path = os.path.join(self.log_dir, f"{name}.jpg")
            imageio.imwrite(image_path, np.uint8(image.cpu().numpy() * 255.0))

    def write_scalar(self, name: str, scalar: float, step: int) -> None:
        if step > 0:
            latest_map, new_key = self._consolidate_events()
            self._update_header(latest_map, new_key)
            self._print_stats(latest_map)

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
        """helper to handle the printing of the header labels"""
        full_log_cond = not self.max_log_size and GLOBAL_BUFFER["step"] <= GLOBAL_BUFFER["steps_per_log"]
        capped_log_cond = self.max_log_size and (len(self.past_mssgs) <= 2 or new_key)
        if full_log_cond or capped_log_cond:
            mssg = f"{'Step (% Done)':<20}"
            for name, _ in latest_map.items():
                if name in self.stats_to_track:
                    mssg += f"{name:<20} "
            self.past_mssgs[0] = mssg
            self.past_mssgs[1] = "-" * len(mssg)
            if full_log_cond:
                print(mssg)
                print("-" * len(mssg))

    def _print_stats(self, latest_map, padding=" "):
        """helper to print out the stats in a readable format"""
        step = GLOBAL_BUFFER["step"]
        fraction_done = step / GLOBAL_BUFFER["max_iter"]
        curr_mssg = f"{step} ({fraction_done*100:.02f}%)"
        curr_mssg = f"{curr_mssg:<20}"
        for name, v in latest_map.items():
            if name in self.stats_to_track:
                if "(time)" in name:
                    v = _format_time(v)
                else:
                    v = f"{v:0.4f}"
                curr_mssg += f"{v:<20} "

        # update the history buffer
        if self.max_log_size:
            cursor_idx = len(self.past_mssgs)
            if len(self.past_mssgs[2:]) >= self.max_log_size:
                self.past_mssgs.pop(2)
            self.past_mssgs.append(curr_mssg)
            _cursorup(cursor_idx)

            for mssg in self.past_mssgs:
                pad_len = len(self.past_mssgs[0])
                print(f"{mssg:{padding}<{pad_len}}")
        else:
            print(curr_mssg)
