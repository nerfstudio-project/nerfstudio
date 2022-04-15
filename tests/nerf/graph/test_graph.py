"""
Tests of graph construction
"""
import numpy as np
from hydra import compose, initialize
from omegaconf import open_dict

from mattport.nerf.graph import Graph, Node

CONFIG_DIR = "configs"


def check_consistency(root: Node, targets: list):
    """rudimentary check on predicted graph versus expected target ordering"""
    curr_pointer = root
    for i, (target_name, target_children) in enumerate(targets):
        assert curr_pointer.name == target_name
        assert len(curr_pointer.children) == target_children
        if len(curr_pointer.children) > 0:
            assert targets[i + 1][0] in curr_pointer.children
            curr_pointer = curr_pointer.children[targets[i + 1][0]]


def instantiate_graph(config_name: str) -> Graph:
    """load correct graph given hydra config"""
    with initialize(config_path=CONFIG_DIR):
        cfg = compose(config_name=config_name)
    test_graph = Graph(cfg.network)
    return test_graph, cfg


def test_graph_init():
    """test field modules dictionary is instantiated properly"""
    test_graph, _ = instantiate_graph("test_basic_graph")
    assert test_graph is not None
    assert type(test_graph["encoder_0"]).__name__ == "ScalingAndOffset"
    assert type(test_graph["mlp_0"]).__name__ == "MLP"
    assert type(test_graph["mlp_1"]).__name__ == "MLP"


def test_input_dimension():
    """test field module input dimension calculation"""
    test_graph, _ = instantiate_graph("test_repeat_graph")
    assert test_graph["mlp_0"].in_dim == 16
    assert test_graph["mlp_1"].in_dim == 1


def test_construct_graph_basic():
    """test construction of dependency graph: ensure correct ordering"""
    test_graph, _ = instantiate_graph("test_basic_graph")
    roots = test_graph.construct_graph()
    assert len(roots) == 1
    roots = list(roots)
    targets = [("encoder_0", 1), ("mlp_0", 1), ("mlp_1", 0)]
    check_consistency(roots[0], targets)


def test_construct_graph_repeat():
    """test construction of dependency graph: ensure duplicate nodes are ok"""
    test_graph, cfg = instantiate_graph("test_repeat_graph")
    roots = test_graph.construct_graph()
    assert len(roots) == 1
    roots = list(roots)
    targets = [("encoder_0", 1), ("mlp_0", 1), ("mlp_1", 0)]
    check_consistency(roots[0], targets)

    with open_dict(cfg):
        cfg.network.mlp_0.inputs[1] = "encoder_1"
        cfg.network.encoder_1 = {
            "class_name": "ScalingAndOffset",
            "inputs": ["x"],
            "meta_data": {"in_dim": 8},
        }
    test_graph = Graph(cfg.network)
    roots = test_graph.construct_graph()
    assert len(roots) == 2


def test_construct_graph_complex():
    """test construction of dependency graph: ensure different depths are ok"""
    test_graph, _ = instantiate_graph("test_complex_graph")
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
    test_basic_graph, _ = instantiate_graph("test_basic_graph")
    _ = test_basic_graph.construct_graph()
    order = test_basic_graph.module_order
    target = ["encoder_0", "mlp_0", "mlp_1"]
    assert np.array_equal(target, order)

    test_repeated_graph, cfg = instantiate_graph("test_repeat_graph")
    _ = test_repeated_graph.construct_graph()
    order = test_repeated_graph.module_order
    target = ["encoder_0", "mlp_0", "mlp_1"]
    assert np.array_equal(target, order)

    with open_dict(cfg):
        cfg.network.mlp_0.inputs[1] = "encoder_1"
        cfg.network.encoder_1 = {
            "class_name": "ScalingAndOffset",
            "inputs": ["x"],
            "meta_data": {"in_dim": 16},
        }
    test_repeated_graph = Graph(cfg.network)
    order = test_repeated_graph.module_order
    assert order.index("mlp_0") > order.index("encoder_0") and order.index("mlp_0") > order.index("encoder_1")

    test_complex_graph, _ = instantiate_graph("test_complex_graph")
    _ = test_complex_graph.construct_graph()
    order = test_complex_graph.module_order
    assert order.index("mlp_2") > order.index("mlp_0")
    assert order.index("mlp_2") > order.index("encoder_0")
    assert order.index("mlp_3") > order.index("encoder_0")
