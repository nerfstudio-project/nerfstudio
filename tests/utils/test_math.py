import pytest
import torch

from nerfstudio.utils.math import components_from_spherical_harmonics


@pytest.mark.parametrize("components", list(range(1, 5 + 1)))
def test_spherical_harmonics(components):
    torch.manual_seed(0)
    N = 1000000

    dx = torch.normal(0, 1, size=(N, 3))
    dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True)
    sh = components_from_spherical_harmonics(components, dx)
    matrix = (sh.T @ sh) / N * 4 * torch.pi
    torch.testing.assert_close(matrix, torch.eye(components**2), rtol=0, atol=1.5e-2)
