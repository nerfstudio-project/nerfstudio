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
Module that keeps all registered plugins and allows for plugin discovery.
"""

import sys
import typing as t

from rich.progress import Console

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
CONSOLE = Console(width=120)


def discover_methods() -> t.Tuple[t.Dict[str, TrainerConfig], t.Dict[str, str]]:
    """
    Discovers all methods registered using the `nerfstudio.method_configs` entrypoint.
    """
    methods = {}
    descriptions = {}
    discovered_entry_points = entry_points(group="nerfstudio.method_configs")
    for name in discovered_entry_points.names:
        specification = discovered_entry_points[name].load()
        if not isinstance(specification, MethodSpecification):
            CONSOLE.print(
                "[bold yellow]Warning: Could not entry point {n} as it is not an instance of MethodSpecification"
            )
            continue
        specification = t.cast(MethodSpecification, specification)
        methods[specification.config.method_name] = specification.config
        descriptions[specification.config.method_name] = specification.description
    return methods, descriptions
