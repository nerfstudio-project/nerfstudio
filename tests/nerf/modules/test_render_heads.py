"""
Render Head Tests
"""
from mattport.nerf.modules import render_heads


def test_density_head():
    """Test rgb render head"""
    in_dim = 6
    density_head = render_heads.DensityHead(in_dim)
    assert density_head.get_out_dim() == 1


def test_rgb_head():
    """Test rgb render head"""
    in_dim = 6
    rgb_head = render_heads.RGBHead(in_dim)
    assert rgb_head.get_out_dim() == 3
