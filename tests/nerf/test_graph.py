import pytest

from mattport.nerf.graph import Graph


def create_basic_graph():
    """Create basic test for creating a graph

    Returns:
        dict: module definitions for constructing a graph
    """
    basic_graph = {
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
    return basic_graph


def test_graph_init():
    """test modules dictionary is instantiated properly"""
    test_graph = Graph(create_basic_graph())
    assert test_graph.modules is not None
    assert type(test_graph.modules['encoder_0']).__name__ == 'Encoding'
    assert type(test_graph.modules['mlp_0']).__name__ == 'MLP'
    assert type(test_graph.modules['mlp_1']).__name__ == 'MLP'


def test_input_dimension():
    """test module input dimension calculation"""
    test_graph = Graph(create_basic_graph())
    assert not hasattr(test_graph.modules['encoder_0'], 'in_dim')
    assert test_graph.modules['mlp_0'].in_dim == 16
    assert test_graph.modules['mlp_1'].in_dim == 1
    