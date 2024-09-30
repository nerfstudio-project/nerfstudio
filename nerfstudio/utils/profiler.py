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

"""
Profiler base class and functionality
"""

from __future__ import annotations

import functools
import os
import time
import typing
from collections import deque
from contextlib import ContextDecorator, contextmanager
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, TypeVar, Union, overload

from torch.profiler import ProfilerActivity, profile, record_function

from nerfstudio.configs import base_config as cfg
from nerfstudio.utils import comms
from nerfstudio.utils.decorators import check_main_thread, check_profiler_enabled, decorate_all
from nerfstudio.utils.rich_utils import CONSOLE

PROFILER = []
PYTORCH_PROFILER = None


CallableT = TypeVar("CallableT", bound=Callable)


@overload
def time_function(name_or_func: CallableT) -> CallableT: ...


@overload
def time_function(name_or_func: str) -> ContextManager[Any]: ...


def time_function(name_or_func: Union[CallableT, str]) -> Union[CallableT, ContextManager[Any]]:
    """Profile a function or block of code. Can be used either to create a context or to wrap a function.

    Args:
        name_or_func: Either the name of a context or function to profile.

    Returns:
        A wrapped function or context to use in a `with` statement.
    """
    return _TimeFunction(name_or_func)


class _TimeFunction(ContextDecorator):
    """Decorator/Context manager: time a function call or a block of code"""

    def __init__(self, name: Union[str, Callable]):
        # NOTE: This is a workaround for the fact that the __new__ method of a ContextDecorator
        # is not picked up by VSCode intellisense
        self.name: str = typing.cast(str, name)
        self.start = None
        self._profiler_contexts = deque()
        self._function_call_args: Optional[Tuple[Tuple, Dict]] = None

    def __new__(cls, func: Union[str, Callable]):
        instance = super().__new__(cls)
        if isinstance(func, str):
            instance.__init__(func)
            return instance
        if callable(func):
            instance.__init__(func.__qualname__)
            return instance(func)
        raise ValueError(f"Argument func of type {type(func)} is not a string or a callable.")

    def __enter__(self):
        self.start = time.time()
        if PYTORCH_PROFILER is not None:
            args, kwargs = tuple(), {}
            if self._function_call_args is not None:
                args, kwargs = self._function_call_args
            ctx = PYTORCH_PROFILER.record_function(self.name, *args, **kwargs)
            ctx.__enter__()
            self._profiler_contexts.append(ctx)
            if self._function_call_args is None:
                ctx = record_function(self.name)
                ctx.__enter__()
                self._profiler_contexts.append(ctx)

    def __exit__(self, *args, **kwargs):
        while self._profiler_contexts:
            context = self._profiler_contexts.pop()
            context.__exit__(*args, **kwargs)
        if PROFILER:
            PROFILER[0].update_time(self.name, self.start, time.time())

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            self._function_call_args = (args, kwargs)
            with self:
                out = func(*args, **kwargs)
            self._function_call_args = None
            return out

        return inner


def flush_profiler(config: cfg.LoggingConfig):
    """Method that checks if profiler is enabled before flushing"""
    if config.profiler != "none" and PROFILER:
        PROFILER[0].print_profile()


def setup_profiler(config: cfg.LoggingConfig, log_dir: Path):
    """Initialization of profilers"""
    global PYTORCH_PROFILER
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
            # Some arbitrary steps which likely do not overlap with steps usually chosen to run callbacks
            trace_steps = [12, 17]
        self.trace_steps = trace_steps

    @contextmanager
    def record_function(self, function: str, *args, **_kwargs):
        """
        Context manager that records a function call and saves the trace to a json file.
        Traced functions are: train_iteration, eval_iteration
        """
        if function.endswith("train_iteration") or function.endswith("eval_iteration"):
            step = args[1]
            assert isinstance(step, int)
            assert len(args) == 2
            stage = function.split(".")[-1].split("_")[0]
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
                    yield None
                if launch_kernel_blocking:
                    os.environ["CUDA_LAUNCH_BLOCKING"] = backup_lb_var
                self.output_path.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(
                    str(self.output_path / f"trace_{stage}_{step}{'_blocking' if launch_kernel_blocking else ''}.json")
                )
                return
        # Functions are recorded automatically
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
