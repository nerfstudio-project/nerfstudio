"""
Field output tests
"""
import pytest
import torch
from torch import nn

from mattport.nerf.field_modules import field_heads


def test_field_output():
    """Test render output"""
    in_dim = 6
    out_dim = 4
    activation = nn.ReLU()
    render_head = field_heads.FieldHead(
        in_dim=in_dim, out_dim=out_dim, field_quantity_name="rgb", activation=activation
    )
    assert render_head.get_out_dim() == out_dim

    x = torch.ones((9, in_dim))
    y = render_head(x)


def test_density_output():
    """Test rgb output"""
    in_dim = 6
    density_head = field_heads.DensityFieldHead(in_dim)
    assert density_head.get_out_dim() == 1

    x = torch.ones((9, in_dim))
    y = density_head(x)


def test_rgb_output():
    """Test rgb output"""
    in_dim = 6
    rgb_head = field_heads.RGBFieldHead(in_dim)
    assert rgb_head.get_out_dim() == 3

    x = torch.ones((9, in_dim))
    y = rgb_head(x)
