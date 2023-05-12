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


def discover_dataparsers() -> t.Dict[str, DataParserConfig]:
    """
    Discovers all dataparsers registered using the `nerfstudio.dataparser_configs` entrypoint.
    """
    dataparsers = {}
    discovered_entry_points = entry_points(group="nerfstudio.dataparser_configs")
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        if not isinstance(spec, DataParserSpecification):
            CONSOLE.print(
                f"[bold yellow]Warning: Could not entry point {spec} as it is an instance of DataParserSpecification"
            )
            continue
        spec = t.cast(DataParserSpecification, spec)
        dataparsers[name] = spec.config

    return dataparsers
