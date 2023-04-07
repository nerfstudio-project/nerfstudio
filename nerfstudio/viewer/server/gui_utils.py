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

""" Utilites for generating custom gui elements in the viewer """

from __future__ import annotations

from typing import List, Tuple

from torch import nn

from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.viewer.server.viewer_param import ViewerElement


def parse_object(
    obj,
    type_check,
    tree_stub,
):
    if not hasattr(obj, "__dict__"):
        return []
    ret = []
    for k in dir(obj):
        if k[0] == "_":
            continue
        v = getattr(obj, k)
        new_tree_stub = f"{tree_stub}/{k}"
        if isinstance(v, type_check):
            ret.append((new_tree_stub, v))
        elif isinstance(v, nn.Module):
            ret.extend(parse_object(v, type_check, new_tree_stub))
    return ret


def get_viewer_elements(pipeline: Pipeline) -> List[Tuple[str, ViewerElement]]:
    """
    Recursively parse the pipeline object and return a tree of all the ViewerElements contained

    returns a list of (path/to/object, param), which represents the path down the object tree
    """
    ret = parse_object(pipeline, ViewerElement, "Custom")
    return ret
