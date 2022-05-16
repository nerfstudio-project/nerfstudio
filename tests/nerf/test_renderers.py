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


def test_sh_renderer():
    """Test SH volumetric rendering"""

    levels = 4
    num_samples = 10

    sh = torch.ones((3, num_samples, 3 * levels**2))
    weights = torch.ones((3, num_samples))
    weights /= torch.sum(weights, axis=-1, keepdim=True)
    directions = torch.zeros((3, num_samples, 3))
    directions[..., 0] = 1

    sh_renderer = renderers.SHRenderer()

    out = sh_renderer(sh=sh, directions=directions, weights=weights)
    assert torch.max(out.rgb) > 0.9


def test_acc_renderer():
    """Test accumulation rendering"""

    num_samples = 10
    weights = torch.ones((3, num_samples))
    weights /= torch.sum(weights, axis=-1, keepdim=True)

    acc_renderer = renderers.AccumulationRenderer()

    out = acc_renderer(weights=weights)
    assert torch.max(out.accumulation) > 0.9


def test_depth_renderer():
    """Test depth rendering"""

    num_samples = 10
    weights = torch.ones((3, num_samples))
    weights /= torch.sum(weights, axis=-1, keepdim=True)

    ts = torch.linspace(0, 100, num_samples)
    ts = torch.stack([ts, ts, ts], dim=0)

    depth_renderer = renderers.DepthRenderer()

    out = depth_renderer(weights=weights, ts=ts)
    assert torch.min(out.depth) > 0


if __name__ == "__main__":
    test_rgb_renderer()
    test_sh_renderer()
    test_acc_renderer()
    test_depth_renderer()
