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
