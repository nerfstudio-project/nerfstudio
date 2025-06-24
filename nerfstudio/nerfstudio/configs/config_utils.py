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
Some utility code for configs.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict

from nerfstudio.utils.rich_utils import CONSOLE


# cannot use mutable types directly within dataclass; abstracting default factory calls
def to_immutable_dict(d: Dict[str, Any]):
    """Method to convert mutable dict to default factory dict

    Args:
        d: dictionary to convert into default factory dict for dataclass
    """
    return field(default_factory=lambda: dict(d))


def convert_markup_to_ansi(markup_string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        markup_string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with CONSOLE.capture() as out:
        CONSOLE.print(markup_string, soft_wrap=True)
    return out.get()
