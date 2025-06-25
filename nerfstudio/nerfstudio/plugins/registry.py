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
Module that keeps all registered plugins and allows for plugin discovery.
"""

import importlib
import os
import sys
import typing as t

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.utils.rich_utils import CONSOLE

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


def discover_methods() -> t.Tuple[t.Dict[str, TrainerConfig], t.Dict[str, str]]:
    """
    Discovers all methods registered using the `nerfstudio.method_configs` entrypoint.
    And also methods in the NERFSTUDIO_METHOD_CONFIGS environment variable.
    """
    methods = {}
    descriptions = {}
    discovered_entry_points = entry_points(group="nerfstudio.method_configs")
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        if not isinstance(spec, MethodSpecification):
            CONSOLE.print(
                f"[bold yellow]Warning: Could not entry point {spec} as it is not an instance of MethodSpecification"
            )
            continue
        spec = t.cast(MethodSpecification, spec)
        methods[spec.config.method_name] = spec.config
        descriptions[spec.config.method_name] = spec.description

    if "NERFSTUDIO_METHOD_CONFIGS" in os.environ:
        try:
            strings = os.environ["NERFSTUDIO_METHOD_CONFIGS"].split(",")
            for definition in strings:
                if not definition:
                    continue
                name, path = definition.split("=")
                CONSOLE.print(f"[bold green]Info: Loading method {name} from environment variable")
                module, config_name = path.split(":")
                method_config = getattr(importlib.import_module(module), config_name)

                # method_config specified as function or class -> instance
                if callable(method_config):
                    method_config = method_config()

                # check for valid instance type
                if not isinstance(method_config, MethodSpecification):
                    raise TypeError("Method is not an instance of MethodSpecification")

                # save to methods
                methods[name] = method_config.config
                descriptions[name] = method_config.description
        except Exception:
            CONSOLE.print_exception()
            CONSOLE.print("[bold red]Error: Could not load methods from environment variable NERFSTUDIO_METHOD_CONFIGS")

    return methods, descriptions
