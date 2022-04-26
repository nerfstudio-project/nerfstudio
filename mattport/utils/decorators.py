"""
Decorator definitions
"""
from typing import Callable


def check_main_thread(func: Callable) -> Callable:
    """Declarator: check if you are on main thread"""

    def wrapper(self, *args, **kwargs):
        if self.is_main_thread:
            func(self, *args, **kwargs)

    return wrapper


def check_print_stats_step(func: Callable) -> Callable:
    """Declarator: check if it is time to print stats update"""

    def wrapper(self, *args, **kwargs):
        if (
            self.step % self.config.steps_per_log == 0
            or (self.config.steps_per_save and self.step % self.config.steps_per_save == 0)
            or self.step % self.config.steps_per_test == 0
        ):
            func(self, *args, **kwargs)

    return wrapper
