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
Profiler base class and functionality
"""
from __future__ import annotations

import time
from typing import Callable

from rich.console import Console

from nerfstudio.configs import base_config as cfg
from nerfstudio.utils import comms
from nerfstudio.utils.decorators import (
    check_main_thread,
    check_profiler_enabled,
    decorate_all,
)

CONSOLE = Console(width=120)

PROFILER = []


def time_function(func: Callable) -> Callable:
    """Decorator: time a function call"""

    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        if PROFILER:
            class_str = func.__qualname__
            PROFILER[0].update_time(class_str, start, time.time())
        return ret

    return wrapper


def flush_profiler(config: cfg.LoggingConfig):
    """Method that checks if profiler is enabled before flushing"""
    if config.enable_profiler and PROFILER:
        PROFILER[0].print_profile()


def setup_profiler(config: cfg.LoggingConfig):
    """Initialization of profilers"""
    if comms.is_main_process():
        PROFILER.append(Profiler(config))


@decorate_all([check_profiler_enabled, check_main_thread])
class Profiler:
    """Profiler class"""

    def __init__(self, config: cfg.LoggingConfig):
        self.config = config
        self.profiler_dict = {}

    def update_time(self, func_name: str, start_time: float, end_time: float):
        """update the profiler dictionary with running averages of durations

        Args:
            func_name: the function name that is being profiled
            start_time: the start time when function is called
            end_time: the end time when function terminated
        """
        val = end_time - start_time
        func_dict = self.profiler_dict.get(func_name, {"val": 0, "step": 0})
        prev_val = func_dict["val"]
        prev_step = func_dict["step"]
        self.profiler_dict[func_name] = {"val": (prev_val * prev_step + val) / (prev_step + 1), "step": prev_step + 1}

    def print_profile(self):
        """helper to print out the profiler stats"""
        CONSOLE.print("Printing profiling stats, from longest to shortest duration in seconds")
        sorted_keys = sorted(
            self.profiler_dict.keys(),
            key=lambda k: self.profiler_dict[k]["val"],
            reverse=True,
        )
        for k in sorted_keys:
            val = f"{self.profiler_dict[k]['val']:0.4f}"
            CONSOLE.print(f"{k:<20}: {val:<20}")
