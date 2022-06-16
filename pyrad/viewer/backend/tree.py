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

from collections import defaultdict


class TreeNode(defaultdict):
    __slots__ = ["object", "transform", "properties", "animation"]

    def __init__(self, *args, **kwargs):
        super(TreeNode, self).__init__(*args, **kwargs)
        self.object = None
        self.properties = []
        self.transform = None
        self.animation = None


SceneTree = lambda: TreeNode(SceneTree)


def walk(tree):
    yield tree
    for v in tree.values():
        for t in walk(v):  # could use `yield from` if we didn't need python2
            yield t


def find_node(tree, path):
    if len(path) == 0:
        return tree
    else:
        return find_node(tree[path[0]], path[1:])
