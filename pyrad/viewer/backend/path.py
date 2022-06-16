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

from __future__ import absolute_import, division, print_function

import sys

if sys.version_info >= (3, 0):
    unicode = str


class Path(object):
    __slots__ = ["entries"]

    def __init__(self, entries=tuple()):
        self.entries = entries

    def append(self, other):
        new_path = self.entries
        for element in other.split("/"):
            if len(element) == 0:
                new_path = tuple()
            else:
                new_path = new_path + (element,)
        return Path(new_path)

    def lower(self):
        return unicode("/" + "/".join(self.entries))

    def __hash__(self):
        return hash(self.entries)

    def __eq__(self, other):
        return self.entries == other.entries
