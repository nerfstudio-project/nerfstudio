"""
Render Head Tests
"""
import pytest
import torch
from torch import nn

from mattport.nerf.field_modules import field_heads


def test_field_output():
    """Test rgb render head"""
    in_dim = 6
    out_dim = 4
    activation = nn.ReLU()
    render_head = field_heads.FieldHead(in_dim=in_dim, out_dim=out_dim, activation=activation)
    assert render_head.get_out_dim() == out_dim

    x = torch.ones((9, in_dim))

    with pytest.raises(SystemError):
        render_head(x)
    render_head.build_nn_modules()
    y = render_head(x)

    assert y.shape[-1] == out_dim


def test_density_head():
    """Test rgb output"""
    in_dim = 6
    density_head = field_heads.DensityFieldHead(in_dim)
    assert density_head.get_out_dim() == 1

    x = torch.ones((9, in_dim))
    with pytest.raises(SystemError):
        density_head(x)
    density_head.build_nn_modules()
    y = density_head(x)

    assert y.shape[-1] == 1


def test_rgb_head():
    """Test rgb output"""
    in_dim = 6
    rgb_head = field_heads.RGBFieldHead(in_dim)
    assert rgb_head.get_out_dim() == 3

    x = torch.ones((9, in_dim))
    with pytest.raises(SystemError):
        rgb_head(x)
    rgb_head.build_nn_modules()
    y = rgb_head(x)

    assert y.shape[-1] == 3
