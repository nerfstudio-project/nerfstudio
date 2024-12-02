import pytest
import torch

from nerfstudio.utils.spherical_harmonics import components_from_spherical_harmonics, num_sh_bases


@pytest.mark.parametrize("degree", list(range(0, 5)))
def test_spherical_harmonics(degree):
    torch.manual_seed(0)
    N = 1000000

    dx = torch.normal(0, 1, size=(N, 3))
    dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True)
    sh = components_from_spherical_harmonics(degree, dx)
    matrix = (sh.T @ sh) / N * 4 * torch.pi
    torch.testing.assert_close(matrix, torch.eye(num_sh_bases(degree)), rtol=0, atol=1.5e-2)
