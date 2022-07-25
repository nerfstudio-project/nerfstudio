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
Functions for traversing server bridge tree nodes (storage)
"""

from __future__ import absolute_import, division, print_function

from collections import defaultdict
from typing import List


class TreeNode(defaultdict):
    """Tree Node object"""

    __slots__ = ["object", "transform", "properties", "animation"]

    def __init__(self, *args, **kwargs):
        super(TreeNode, self).__init__(*args, **kwargs)
        self.object = None
        self.properties = []
        self.transform = None
        self.animation = None


SceneTree = lambda: TreeNode(SceneTree)


def walk(tree: TreeNode) -> None:
    """Walk the entire tree and return the values

    Args:
        tree: the root of the tree to start search
    """
    yield tree
    for v in tree.values():
        for t in walk(v):  # could use `yield from` if we didn't need python2
            yield t


def find_node(tree: TreeNode, path: List[str]) -> TreeNode:
    """Find the node associated with the path

    Args:
        tree: the root of the tree to start search
        path: the path we are searching for
    """
    if len(path) == 0:
        return tree
    else:
        return find_node(tree[path[0]], path[1:])
