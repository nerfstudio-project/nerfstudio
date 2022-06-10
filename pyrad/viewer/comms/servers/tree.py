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
