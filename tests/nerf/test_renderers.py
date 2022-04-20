"""
Test renderers
"""
import pytest
import torch

from mattport.nerf import renderers


def test_rgb_renderer():
    """Test RGB volumetric rendering"""
    num_samples = 10

    rgb = torch.ones((3, num_samples, 3))
    densities = torch.ones((3, num_samples, 1))
    deltas = torch.ones((3, num_samples))

    rgb_renderer = renderers.RGBRenderer()

    assert rgb_renderer.required_field_outputs is not None

    out = rgb_renderer(rgb=rgb, density=densities, deltas=deltas)
    assert torch.max(out.rgb) > 0.9

    out = rgb_renderer(rgb=rgb * 0, density=densities, deltas=deltas)
    assert torch.max(out.rgb) == pytest.approx(0)

    out = rgb_renderer(rgb=rgb, density=densities * 0, deltas=deltas)
    assert torch.max(out.rgb) == pytest.approx(0)

    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_rgb_renderer()
