import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScR

from nerfstudio.utils.spherical_harmonics import (
    components_from_spherical_harmonics,
    num_sh_bases,
    rotate_spherical_harmonics,
)


@pytest.mark.parametrize("degree", list(range(0, 5)))
def test_spherical_harmonics(degree):
    torch.manual_seed(0)
    N = 1000000

    dx = torch.normal(0, 1, size=(N, 3))
    dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True)
    sh = components_from_spherical_harmonics(degree, dx)
    matrix = (sh.T @ sh) / N * 4 * torch.pi
    torch.testing.assert_close(matrix, torch.eye(num_sh_bases(degree)), rtol=0, atol=1.5e-2)


@pytest.mark.parametrize("sh_degree", list(range(0, 4)))
def test_rotate_spherical_harmonics(sh_degree):
    torch.manual_seed(0)
    np.random.seed(0)

    num_coeffs = (sh_degree + 1) ** 2
    coeffs = torch.rand(1000, 3, num_coeffs)

    rot_matrix = torch.tensor(ScR.random().as_matrix(), dtype=torch.float32)
    rotated_coeffs = rotate_spherical_harmonics(coeffs, rot_matrix)

    # Norm Preservation
    norm_original = torch.norm(coeffs, dim=-1)
    norm_rotated = torch.norm(rotated_coeffs, dim=-1)
    torch.testing.assert_close(norm_original, norm_rotated, rtol=0, atol=1e-5)

    # 0th order coeffs are invariant to rotation
    torch.testing.assert_close(coeffs[..., 0], rotated_coeffs[..., 0], rtol=0, atol=1e-6)

    # 1st order coeffs undergo simple rotation
    if sh_degree > 0:
        act_1st_order = torch.einsum("ij,...j->...i", rot_matrix, coeffs[..., 1:4])
        torch.testing.assert_close(act_1st_order, rotated_coeffs[..., 1:4], rtol=0, atol=1e-5)

    # Identity Rotation
    rot_matrix = torch.eye(3)
    rotated_coeffs = rotate_spherical_harmonics(coeffs, rot_matrix)
    torch.testing.assert_close(coeffs, rotated_coeffs, rtol=0, atol=1e-6)
