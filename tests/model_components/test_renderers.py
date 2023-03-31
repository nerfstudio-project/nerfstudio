"""
Test renderers
"""
import pytest
import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.model_components import renderers


def test_rgb_renderer():
    """Test RGB volumetric rendering"""
    num_samples = 10

    rgb_samples = torch.ones((3, num_samples, 3))
    weights = torch.ones((3, num_samples, 1))
    weights /= torch.sum(weights, dim=-2, keepdim=True)

    rgb_renderer = renderers.RGBRenderer()

    rgb = rgb_renderer(rgb=rgb_samples, weights=weights)
    assert torch.max(rgb) > 0.9

    rgb = rgb_renderer(rgb=rgb_samples * 0, weights=weights)
    assert torch.max(rgb) == pytest.approx(0, abs=1e-6)


def test_sh_renderer():
    """Test SH volumetric rendering"""

    levels = 4
    num_samples = 10

    sh = torch.ones((3, num_samples, 3 * levels**2))
    weights = torch.ones((3, num_samples, 1))
    weights /= torch.sum(weights, dim=-2, keepdim=True)
    directions = torch.zeros((3, num_samples, 3))
    directions[..., 0] = 1

    sh_renderer = renderers.SHRenderer()

    rgb = sh_renderer(sh=sh, directions=directions, weights=weights)
    assert torch.max(rgb) > 0.7


def test_acc_renderer():
    """Test accumulation rendering"""

    num_samples = 10
    weights = torch.ones((3, num_samples, 1))
    weights /= torch.sum(weights, dim=-2, keepdim=True)

    acc_renderer = renderers.AccumulationRenderer()

    accumulation = acc_renderer(weights=weights)
    assert torch.max(accumulation) > 0.9


def test_depth_renderer():
    """Test depth rendering"""

    num_samples = 10
    weights = torch.ones((num_samples, 1))
    weights /= torch.sum(weights, dim=-2, keepdim=True)

    frustums = Frustums.get_mock_frustum()
    frustums.starts = torch.linspace(0, 100, num_samples)[..., None]
    frustums.ends = torch.linspace(1, 101, num_samples)[..., None]

    ray_samples = RaySamples(
        frustums=frustums,
        camera_indices=torch.ones((num_samples, 1)),
        deltas=torch.ones((num_samples, 1)),
    )

    depth_renderer = renderers.DepthRenderer(method="median")
    depth = depth_renderer(weights=weights, ray_samples=ray_samples)
    assert torch.min(depth) > 0

    depth_renderer = renderers.DepthRenderer(method="expected")
    depth = depth_renderer(weights=weights, ray_samples=ray_samples)
    assert torch.min(depth) > 0


if __name__ == "__main__":
    test_rgb_renderer()
    test_sh_renderer()
    test_acc_renderer()
    test_depth_renderer()
