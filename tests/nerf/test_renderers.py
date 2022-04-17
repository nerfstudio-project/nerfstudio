"""
Test renderers
"""
import pytest
import torch

from mattport.nerf import renderers


def test_rgb_renderer():
    """Test RGB volumetric rendering"""
    rgb_name = "rgb"
    density_name = "density"
    num_samples = 10

    rgb = torch.ones((3, num_samples, 3))
    densities = torch.ones((3, num_samples, 1))
    deltas = torch.ones((3, num_samples))

    rgb_renderer = renderers.RGB(rgb_name, density_name)

    assert rgb_renderer.required_field_outputs is not None

    field_output_dict = {rgb_name: rgb, density_name: densities}
    out = rgb_renderer(field_output_dict, deltas)
    assert torch.max(out) > 0.9

    field_output_dict = {rgb_name: rgb * 0, density_name: densities}
    out = rgb_renderer(field_output_dict, deltas)
    assert torch.max(out) == pytest.approx(0)

    field_output_dict = {rgb_name: rgb, density_name: densities * 0}
    out = rgb_renderer(field_output_dict, deltas)
    assert torch.max(out) == pytest.approx(0)

    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_rgb_renderer()
