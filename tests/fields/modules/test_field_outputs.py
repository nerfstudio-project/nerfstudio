"""
Field output tests
"""
import pytest
import torch
from torch import nn

from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
    SHFieldHead,
)


def test_field_output():
    """Test render output"""
    in_dim = 6
    out_dim = 4
    activation = nn.ReLU()
    render_head = FieldHead(in_dim=in_dim, out_dim=out_dim, field_head_name=FieldHeadNames.RGB, activation=activation)
    assert render_head.get_out_dim() == out_dim

    x = torch.ones((9, in_dim))
    render_head(x)

    # Test in_dim not provided at construction
    render_head = FieldHead(out_dim=out_dim, field_head_name=FieldHeadNames.RGB, activation=activation)
    with pytest.raises(SystemError):
        render_head(x)
    render_head.set_in_dim(in_dim)
    render_head(x)


def test_density_output():
    """Test rgb output"""
    in_dim = 6
    density_head = DensityFieldHead(in_dim)
    assert density_head.get_out_dim() == 1

    x = torch.ones((9, in_dim))
    density_head(x)


def test_rgb_output():
    """Test rgb output"""
    in_dim = 6
    rgb_head = RGBFieldHead(in_dim)
    assert rgb_head.get_out_dim() == 3

    x = torch.ones((9, in_dim))
    rgb_head(x)


def test_sh_output():
    """Test sh output"""
    in_dim = 6
    levels = 4
    channels = 3
    rgb_head = SHFieldHead(in_dim, levels=levels, channels=channels)
    assert rgb_head.get_out_dim() == channels * levels**2

    x = torch.ones((9, in_dim))
    rgb_head(x)


if __name__ == "__main__":
    test_field_output()
    test_density_output()
    test_rgb_output()
    test_sh_output()
