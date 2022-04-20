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
