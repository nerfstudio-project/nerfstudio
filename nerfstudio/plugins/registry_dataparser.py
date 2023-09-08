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

import os
import importlib
import sys
import typing as t
from dataclasses import dataclass

from rich.progress import Console

from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
CONSOLE = Console(width=120)


@dataclass
class DataParserSpecification:
    """
    DataParser specification class used to register custom dataparsers with Nerfstudio.
    The registered dataparsers will be available in commands such as `ns-train`
    """

    config: DataParserConfig
    """Dataparser configuration"""

    description: t.Optional[str] = None
    """Description of the dataparser"""


def discover_dataparsers() -> t.Tuple[t.Dict[str, DataParserConfig], t.Dict[str, str]]:
    """
    Discovers all dataparsers registered using the `nerfstudio.dataparser_configs` entrypoint.
    And also dataparsers in the NERFSTUDIO_DATAPARSER_CONFIGS environment variable.
    """
    dataparsers = {}
    descriptions = {}
    discovered_entry_points = entry_points(group="nerfstudio.dataparser_configs")
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        if not isinstance(spec, DataParserSpecification):
            CONSOLE.print(
                f"[bold yellow]Warning: Could not entry point {spec} as it is not an instance of DataParserSpecification"
            )
            continue
        spec = t.cast(DataParserSpecification, spec)
        dataparsers[name] = spec.config
        descriptions[name] = spec.description

    if "NERFSTUDIO_DATAPARSER_CONFIGS" in os.environ:
        try:
            strings = os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"].split(",")
            for definition in strings:
                if not definition:
                    continue
                name, path = definition.split("=")
                CONSOLE.print(f"[bold green]Info: Loading method {name} from environment variable")
                module, config_name = path.split(":")
                dataparser_config = getattr(importlib.import_module(module), config_name)

                # method_config specified as function or class -> instance
                if callable(dataparser_config):
                    dataparser_config = dataparser_config()

                # check for valid instance type
                if not isinstance(dataparser_config, DataParserSpecification):
                    raise TypeError("Method is not an instance of DataParserSpecification")

                # save to methods
                dataparsers[name] = dataparser_config.config
                descriptions[name] = dataparser_config.description
        except Exception:  # pylint: disable=broad-except
            CONSOLE.print_exception()
            CONSOLE.print(
                "[bold red]Error: Could not load methods from environment variable NERFSTUDIO_DATAPARSER_CONFIGS"
            )

    return dataparsers, descriptions
