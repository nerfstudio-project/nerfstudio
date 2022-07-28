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
For tree logic code.
"""

from collections import defaultdict
from typing import Callable


class Node(defaultdict):
    """
    The base class Node.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_tree(node_class: Callable) -> Callable:
    """
    Get a tree from a node class.
    This allows one to do tree["path"]["to"]["node"]
    and it will return a new node if it doesn't exist
    or the current node if it does.
    """
    assert isinstance(node_class(), Node)
    tree = lambda: node_class(tree)
    return tree()


def walk(tree):
    """Walk the entire tree and return the values
    Args:
        tree: the root of the tree to start search
    """
    yield tree
    for v in tree.values():
        for t in walk(v):  # could use `yield from` if we didn't need python2
            yield t


def find_node(tree, path):
    if len(path) == 0:
        return tree
    else:
        return find_node(tree[path[0]], path[1:])
