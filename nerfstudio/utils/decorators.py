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
Decorator definitions
"""
from typing import Callable, List

from nerfstudio.utils import comms


def decorate_all(decorators: List[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class

    Args:
        decorators: list of decorators to add to all functions in the class
    """

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != "__init__":
                for decorator in decorators:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def check_profiler_enabled(func: Callable) -> Callable:
    """Decorator: check if profiler is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.enable_profiler:
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_viewer_enabled(func: Callable) -> Callable:
    """Decorator: check if viewer is enabled and only run on main process"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.is_viewer_enabled() and comms.is_main_process():
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_eval_enabled(func: Callable) -> Callable:
    """Decorator: check if evaluation step is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.is_wandb_enabled() or self.config.is_tensorboard_enabled():
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if comms.is_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper
