"""
Module that keeps all registered plugins and allows for plugin discovery.
"""

import sys

from rich.progress import Console

from nerfstudio.engine.trainer import TrainerConfig

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
CONSOLE = Console(width=120)


def discover_methods():
    """
    Discovers all methods registered using the `nerfstudio.method_configs` entrypoint.
    """
    methods = {}
    descriptions = {}
    discovered_entry_points = entry_points(group="nerfstudio.method_configs")
    for name in discovered_entry_points.names:
        method = discovered_entry_points[name].load()
        if not isinstance(method, TrainerConfig):
            CONSOLE.print("[bold yellow]Warning: Could not entry point {n} as it is not an instance of TrainerConfig")
            continue
        methods[method.method_name] = method
        descriptions[method.method_name] = getattr(method, "description", method.method_name)
    return methods, descriptions
