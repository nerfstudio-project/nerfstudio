"""
MLP Test
"""
import torch
from torch import nn

from nerfstudio.field_components import MLP


def test_mlp():
    """Test mlp"""
    in_dim = 6
    out_dim = 10
    num_layers = 2
    layer_width = 32
    out_activation = nn.ReLU()
    mlp = MLP(
        in_dim=in_dim, out_dim=out_dim, num_layers=num_layers, layer_width=layer_width, out_activation=out_activation
    )
    assert mlp.get_out_dim() == out_dim

    x = torch.ones((9, in_dim))

    mlp.build_nn_modules()
    y = mlp(x)

    assert y.shape[-1] == out_dim


if __name__ == "__main__":
    test_mlp()
