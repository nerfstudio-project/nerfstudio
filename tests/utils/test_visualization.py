"""
Test colormaps
"""
import torch

from nerfstudio.utils import colormaps, plotly_utils


def test_apply_colormap():
    """Test adding a colormap to data"""
    data = torch.rand((10, 20, 1))
    colored_data = colormaps.apply_colormap(data)

    assert colored_data.shape == (10, 20, 3)
    assert torch.min(colored_data) >= 0
    assert torch.max(colored_data) <= 1


def test_apply_depth_colormap():
    """Test adding a colormap to depth data"""
    data = torch.rand((10, 20, 1))
    accum = torch.rand((10, 20, 1))
    accum = accum / torch.max(accum)
    colored_data = colormaps.apply_depth_colormap(depth=data, accumulation=accum)

    assert colored_data.shape == (10, 20, 3)
    assert torch.min(colored_data) >= 0
    assert torch.max(colored_data) <= 1


def test_apply_boolean_colormap():
    """Test adding a colormap to boolean data"""
    data = torch.rand((10, 20, 1))
    data = data > 0.5
    colored_data = colormaps.apply_boolean_colormap(data)

    assert colored_data.shape == (10, 20, 3)
    assert torch.min(colored_data) >= 0
    assert torch.max(colored_data) <= 1


def test_cube_center():
    """Test adding a cube"""
    cube = plotly_utils.get_cube(1.0, torch.Tensor([2.0, 3.0, 4.0]))
    assert cube.x is not None
    assert cube.y is not None
    assert cube.z is not None
    assert cube.x[0] == 1.5
    assert cube.y[0] == 2.5
    assert cube.z[-1] == 4.5


def test_aabb_center():
    """Test adding a cube with different dsid length"""
    cube = plotly_utils.get_cube(torch.Tensor([2.0, 1.0, 3.0]))
    assert cube.x is not None
    assert cube.y is not None
    assert cube.z is not None
    assert cube.x[0] == -1
    assert cube.y[0] == -0.5
    assert cube.z[-1] == 1.5
