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

"""Path class
"""


from typing import Tuple

UNICODE = str


class Path:
    """Path class

    Args:
        entries: component parts of the path
    """

    __slots__ = ["entries"]

    def __init__(self, entries: Tuple = tuple()):
        self.entries = entries

    def append(self, other: str) -> "Path":
        """Method that appends a new component and returns new Path

        Args:
            other: _description_
        """
        new_path = self.entries
        for element in other.split("/"):
            if len(element) == 0:
                new_path = tuple()
            else:
                new_path = new_path + (element,)
        return Path(new_path)

    def lower(self):
        """Convert path object to serializable format"""
        return UNICODE("/" + "/".join(self.entries))

    def __hash__(self):
        return hash(self.entries)

    def __eq__(self, other):
        return self.entries == other.entries
