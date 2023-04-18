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

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional

from rich.console import Console
from torch.profiler import ProfilerActivity, profile, record_function

from nerfstudio.configs import base_config as cfg
from nerfstudio.utils import comms
from nerfstudio.utils.decorators import (
    check_main_thread,
    check_profiler_enabled,
    decorate_all,
)

CONSOLE = Console(width=120)

PROFILER = []
PYTORCH_PROFILER = None


def time_function(func: Callable) -> Callable:
    """Decorator: time a function call"""

    def wrapper(*args, **kwargs):
        class_str = func.__qualname__
        start = time.time()
        if PYTORCH_PROFILER is not None:
            with PYTORCH_PROFILER.record_function(class_str, *args, **kwargs):
                ret = func(*args, **kwargs)
        else:
            ret = func(*args, **kwargs)
        if PROFILER:
            PROFILER[0].update_time(class_str, start, time.time())
        return ret

    return wrapper


def flush_profiler(config: cfg.LoggingConfig):
    """Method that checks if profiler is enabled before flushing"""
    if config.profiler != "none" and PROFILER:
        PROFILER[0].print_profile()


def setup_profiler(config: cfg.LoggingConfig, log_dir: Path):
    """Initialization of profilers"""
    global PYTORCH_PROFILER  # pylint: disable=global-statement
    if comms.is_main_process():
        PROFILER.append(Profiler(config))
        if config.profiler == "pytorch":
            PYTORCH_PROFILER = PytorchProfiler(log_dir)


class PytorchProfiler:
    """
    Wrapper for Pytorch Profiler
    """

    def __init__(self, output_path: Path, trace_steps: Optional[List[int]] = None):
        self.output_path = output_path / "profiler_traces"
        if trace_steps is None:
            trace_steps = [12, 17]
        self.trace_steps = trace_steps

    @contextmanager
    def record_function(self, function: str, *args, **_kwargs):
        """
        Context manager that records a function call and saves the trace to a json file.
        Traced functions are: train_iteration, eval_iteration
        """
        if function == "train_iteration" or function == "eval_iteration":
            step = args[0]
            assert isinstance(step, int)
            if step in self.trace_steps:
                launch_kernel_blocking = self.trace_steps.index(step) % 2 == 0
                backup_lb_var = ""
                if launch_kernel_blocking:
                    backup_lb_var = os.environ.get("CUDA_LAUNCH_BLOCKING", "")
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True,
                ) as prof:
                    with record_function(function):
                        yield None
                if launch_kernel_blocking:
                    os.environ["CUDA_LAUNCH_BLOCKING"] = backup_lb_var
                self.output_path.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(
                    str(
                        self.output_path
                        / "traces"
                        / f"trace_{function}_{step}_{'_blocking' if launch_kernel_blocking else ''}.json"
                    )
                )
                return
        with record_function(function):
            yield None
            return


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
