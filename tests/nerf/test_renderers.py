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
    weights = torch.ones((3, num_samples))
    weights /= torch.sum(weights, axis=-1, keepdim=True)

    rgb_renderer = renderers.RGBRenderer()

    out = rgb_renderer(rgb=rgb, weights=weights)
    assert torch.max(out.rgb) > 0.9

    out = rgb_renderer(rgb=rgb * 0, weights=weights)
    assert torch.max(out.rgb) == pytest.approx(0)

    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_rgb_renderer()
