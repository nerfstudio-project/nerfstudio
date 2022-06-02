"""
Decorator definitions
"""
from typing import Callable, List

from pyrad.utils import comms


def decorate_all(decorators: List[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class"""

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


def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if comms.is_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper
