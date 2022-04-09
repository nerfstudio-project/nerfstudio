import numpy as np
import pytest
from mattport.nerf.graph import Graph


def create_basic_graph():
    """Create basic graph structure
        a -> b -> c

    Returns:
        dict: module definitions for constructing a graph
    """
    basic_graph = {
        "mlp_0": {
            "class_name": "MLP",
            "inputs": ["encoder_0"],
            "meta_data": {"in_dim": 0, "out_dim": 1, "num_layers": 2, "layer_width": 3},
        },
        "encoder_0": {"class_name": "Encoding", "inputs": ["x"], "meta_data": {"out_dim": 8}},
        "mlp_1": {
            "class_name": "MLP",
            "inputs": ["mlp_0"],
            "meta_data": {"in_dim": 4, "out_dim": 5, "num_layers": 6, "layer_width": 7},
        },
    }
    return basic_graph


def create_repeat_graph():
    """Create graph with repeating nodes
        a -> b -> c
        a ->

    Returns:
        dict: module definitions for constructing a graph
    """
    repeat_graph = {
        "encoder_0": {"class_name": "Encoding", "inputs": ["x"], "meta_data": {"out_dim": 8}},
        "mlp_0": {
            "class_name": "MLP",
            "inputs": ["encoder_0", "encoder_0"],
            "meta_data": {"in_dim": 0, "out_dim": 1, "num_layers": 2, "layer_width": 3},
        },
        "mlp_1": {
            "class_name": "MLP",
            "inputs": ["mlp_0"],
            "meta_data": {"in_dim": 4, "out_dim": 5, "num_layers": 6, "layer_width": 7},
        },
    }
    return repeat_graph


def create_complex_graph():
    """Create graph with varied depths, splits and re-joins
        a -> b -> d
               -> e
          -> c
          ------>

    Returns:
        dict: module definitions for constructing a graph
    """
    complex_graph = {
        "encoder_0": {"class_name": "Encoding", "inputs": ["x"], "meta_data": {"out_dim": 8}},
        "mlp_0": {
            "class_name": "MLP",
            "inputs": ["encoder_0"],
            "meta_data": {"in_dim": 0, "out_dim": 1, "num_layers": 2, "layer_width": 3},
        },
        "mlp_1": {
            "class_name": "MLP",
            "inputs": ["encoder_0"],
            "meta_data": {"in_dim": 4, "out_dim": 5, "num_layers": 6, "layer_width": 7},
        },
        "mlp_2": {
            "class_name": "MLP",
            "inputs": ["encoder_0", "mlp_0"],
            "meta_data": {"in_dim": 4, "out_dim": 5, "num_layers": 6, "layer_width": 7},
        },
        "mlp_3": {
            "class_name": "MLP",
            "inputs": ["mlp_0"],
            "meta_data": {"in_dim": 4, "out_dim": 5, "num_layers": 6, "layer_width": 7},
        },
    }
    return complex_graph


def check_consistency(root: "Node", targets: list):
    """rudimentary check on predicted graph versus expected target ordering"""
    curr_pointer = root
    for i, (target_name, target_children) in enumerate(targets):
        assert curr_pointer.name == target_name
        assert len(curr_pointer.children) == target_children
        if len(curr_pointer.children) > 0:
            assert targets[i + 1][0] in curr_pointer.children
            curr_pointer = curr_pointer.children[targets[i + 1][0]]


def test_graph_init():
    """test modules dictionary is instantiated properly"""
    test_graph = Graph(create_basic_graph())
    assert test_graph.modules is not None
    assert type(test_graph.modules["encoder_0"]).__name__ == "Encoding"
    assert type(test_graph.modules["mlp_0"]).__name__ == "MLP"
    assert type(test_graph.modules["mlp_1"]).__name__ == "MLP"


def test_input_dimension():
    """test module input dimension calculation"""
    test_graph = Graph(create_repeat_graph())
    assert not hasattr(test_graph.modules["encoder_0"], "in_dim")
    assert test_graph.modules["mlp_0"].in_dim == 16
    assert test_graph.modules["mlp_1"].in_dim == 1


def test_construct_graph_basic():
    """test construction of dependency graph: ensure correct ordering"""
    test_graph = Graph(create_basic_graph())
    roots = test_graph.construct_graph()
    assert len(roots) == 1
    roots = list(roots)
    targets = [("encoder_0", 1), ("mlp_0", 1), ("mlp_1", 0)]
    check_consistency(roots[0], targets)


def test_construct_graph_repeat():
    """test construction of dependency graph: ensure duplicate nodes are ok"""
    test_graph = Graph(create_repeat_graph())
    roots = test_graph.construct_graph()
    assert len(roots) == 1
    roots = list(roots)
    targets = [("encoder_0", 1), ("mlp_0", 1), ("mlp_1", 0)]
    check_consistency(roots[0], targets)

    test_graph.modules_config["mlp_0"]["inputs"][1] = "encoder_1"
    test_graph.modules_config["encoder_1"] = {"class_name": "Encoding", "inputs": ["x"], "meta_data": {"out_dim": 8}}
    roots = test_graph.construct_graph()
    assert len(roots) == 2


def test_construct_graph_complex():
    """test construction of dependency graph: ensure different depths are ok"""
    test_graph = Graph(create_complex_graph())
    roots = test_graph.construct_graph()
    assert len(roots) == 1
    roots = list(roots)
    targets_p0 = [("encoder_0", 3), ("mlp_0", 2), ("mlp_3", 0)]
    check_consistency(roots[0], targets_p0)

    targets_p1 = [("encoder_0", 3), ("mlp_0", 2), ("mlp_2", 0)]
    check_consistency(roots[0], targets_p1)

    targets_p2 = [("encoder_0", 3), ("mlp_1", 0)]
    check_consistency(roots[0], targets_p2)

    targets_p3 = [("encoder_0", 3), ("mlp_2", 0)]
    check_consistency(roots[0], targets_p3)


def test_ordering():
    """test dependency ordering of constructed graph"""
    test_basic_graph = Graph(create_basic_graph())
    _ = test_basic_graph.construct_graph()
    order = test_basic_graph.get_module_order()
    target = ["encoder_0", "mlp_0", "mlp_1"]
    assert np.array_equal(target, order)

    test_repeated_graph = Graph(create_repeat_graph())
    _ = test_repeated_graph.construct_graph()
    order = test_repeated_graph.get_module_order()
    target = ["encoder_0", "mlp_0", "mlp_1"]
    assert np.array_equal(target, order)

    test_repeated_graph.modules_config["mlp_0"]["inputs"][1] = "encoder_1"
    test_repeated_graph.modules_config["encoder_1"] = {"class_name": "Encoding", "inputs": ["x"], "meta_data": {"out_dim": 8}}
    order = test_repeated_graph.get_module_order()
    assert order.index('mlp_0') > order.index('encoder_0') and order.index('mlp_0') > order.index('encoder_1')

    test_complex_graph = Graph(create_complex_graph())
    _ = test_complex_graph.construct_graph()
    order = test_complex_graph.get_module_order()
    assert order.index('mlp_2') > order.index('mlp_0')
    assert order.index('mlp_2') > order.index('encoder_0')
    assert order.index('mlp_3') > order.index('encoder_0')
