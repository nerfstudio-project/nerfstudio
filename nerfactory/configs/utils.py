# Copyright 2022 The Plenoptix Team. All rights reserved.
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
Some utility code for configs.
"""

from __future__ import annotations

import sys
from dataclasses import field
from typing import Any, Dict, TypeVar

import dcargs

T = TypeVar("T")


def cli_from_base_configs(base_library: Dict[str, T]) -> T:
    """Populate an instance of `cls`, where the first positional argument is used to
    select from a library of named base configs. See
    https://brentyi.github.io/dcargs/examples/06_base_configs/ for more details."""

    # Get base configuration name from the first positional argument.
    if len(sys.argv) < 2 or sys.argv[1] not in base_library:
        valid_usages = map(lambda k: f"{sys.argv[0]} {k} --help", base_library.keys())
        raise SystemExit("usage:\n  " + "\n  ".join(valid_usages))

    # Get base configuration from our library, and use it for default CLI parameters.
    default_instance = base_library[sys.argv[1]]

    return dcargs.cli(
        type(default_instance),
        prog=" ".join(sys.argv[:2]),
        args=sys.argv[2:],
        default_instance=default_instance,
        # `avoid_subparsers` will avoid making a subparser for unions when a default is
        # provided; in this case, it simplifies our CLI but makes it less expressive.
        avoid_subparsers=True,
    )


# pylint: disable=import-outside-toplevel

# cannot use mutable types directly within dataclass; abstracting default factory calls
def to_immutable_dict(d: Dict[str, Any]):
    """Method to convert mutable dict to default factory dict

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda: dict(d))
