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

"""Utilities for generating custom gui elements in the viewer"""

from __future__ import annotations

from typing import Any, List, Tuple

from torch import nn


def parse_object(
    obj: Any,
    type_check,
    tree_stub: str,
) -> List[Tuple[str, Any]]:
    """
    obj: the object to parse
    type_check: recursively adds instances of this type to the output
    tree_stub: the path down the object tree to this object

    Returns:
        a list of (path/to/object, obj), which represents the path down the object tree
        along with the object itself
    """

    def add(ret: List[Tuple[str, Any]], ts: str, v: Any):
        """
        helper that adds to ret, and if v exists already keeps the tree stub with
        the shortest path
        """
        for i, (t, o) in enumerate(ret):
            if o == v:
                if len(t.split("/")) > len(ts.split("/")):
                    ret[i] = (ts, v)
                return
        ret.append((ts, v))

    if not hasattr(obj, "__dict__"):
        return []
    ret = []
    # get a list of the properties of the object, sorted by whether things are instances of type_check
    obj_props = [(k, getattr(obj, k)) for k in dir(obj)]
    for k, v in obj_props:
        if k[0] == "_":
            continue
        new_tree_stub = f"{tree_stub}/{k}"
        if isinstance(v, type_check):
            add(ret, new_tree_stub, v)
        elif isinstance(v, nn.Module):
            if v is obj:
                # some nn.Modules might contain infinite references, e.g. consider foo = nn.Module(), foo.bar = foo
                # to stop infinite recursion, we skip such attributes
                continue
            lower_rets = parse_object(v, type_check, new_tree_stub)
            # check that the values aren't already in the tree
            for ts, o in lower_rets:
                add(ret, ts, o)
    return ret
