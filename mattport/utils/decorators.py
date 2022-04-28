"""
Decorator definitions
"""
from typing import Callable, List


def decorate_all(decorators: List[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class"""

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != "__init__":
                for decorator in decorators:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def check_stats_enabled(func: Callable) -> Callable:
    """Decorator: check if stats tracker is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.logging_configs.enable_stats:
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_profiler_enabled(func: Callable) -> Callable:
    """Decorator: check if profiler is enabled"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.config.logging_configs.enable_stats:
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if self.is_main_thread:
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper


def check_print_stats_step(func: Callable) -> Callable:
    """Decorator: check if it is time to print stats update"""

    def wrapper(self, *args, **kwargs):
        ret = None
        if (
            self.step % self.config.steps_per_log == 0
            or (self.config.steps_per_save and self.step % self.config.steps_per_save == 0)
            or self.step % self.config.steps_per_test == 0
        ):
            ret = func(self, *args, **kwargs)
        return ret

    return wrapper
