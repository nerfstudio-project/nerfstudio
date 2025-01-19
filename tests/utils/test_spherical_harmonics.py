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
def test_spherical_harmonics_components(degree):
    torch.manual_seed(0)
    N = 1000000

    dx = torch.normal(0, 1, size=(N, 3))
    dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True)
    sh = components_from_spherical_harmonics(degree, dx)
    matrix = (sh.T @ sh) / N * 4 * torch.pi
    torch.testing.assert_close(matrix, torch.eye(num_sh_bases(degree)), rtol=0, atol=1.5e-2)


@pytest.mark.parametrize("sh_degree", list(range(0, 4)))
def test_spherical_harmonics_rotation(sh_degree):
    """Test if rotating both the view direction and SH coefficients by the same rotation
    produces the same color output as the original.

     In other words, for any rotation R:
         color(dir, coeffs) = color(R @ dir, rotate_sh(R, coeffs))
    """
    torch.manual_seed(0)
    np.random.seed(0)

    N = 1000
    num_coeffs = (sh_degree + 1) ** 2
    sh_coeffs = torch.rand(N, 3, num_coeffs)
    dirs = torch.rand(N, 3)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)

    y_lm = components_from_spherical_harmonics(sh_degree, dirs)
    color_original = (sh_coeffs * y_lm[..., None, :]).sum(dim=-1)

    rot_matrix = torch.tensor(ScR.random().as_matrix(), dtype=torch.float32)
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs)
    dirs_rotated = (rot_matrix @ dirs.T).T
    y_lm_rotated = components_from_spherical_harmonics(sh_degree, dirs_rotated)
    color_rotated = (sh_coeffs_rotated * y_lm_rotated[..., None, :]).sum(dim=-1)

    torch.testing.assert_close(color_original, color_rotated, rtol=0, atol=1e-5)


@pytest.mark.parametrize("sh_degree", list(range(0, 4)))
def test_spherical_harmonics_rotation_properties(sh_degree):
    """Test properties of the SH rotation"""
    torch.manual_seed(0)
    np.random.seed(0)

    N = 1000
    num_coeffs = (sh_degree + 1) ** 2
    sh_coeffs = torch.rand(N, 3, num_coeffs)
    rot_matrix = torch.tensor(ScR.random().as_matrix(), dtype=torch.float32)
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs)

    # Norm preserving
    norm_original = torch.norm(sh_coeffs, dim=-1)
    norm_rotated = torch.norm(sh_coeffs_rotated, dim=-1)
    torch.testing.assert_close(norm_original, norm_rotated, rtol=0, atol=1e-5)

    # 0th degree coeffs are invariant to rotation
    torch.testing.assert_close(sh_coeffs[..., 0], sh_coeffs_rotated[..., 0], rtol=0, atol=1e-6)

    # Identity rotation
    rot_matrix = torch.eye(3)
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs)
    torch.testing.assert_close(sh_coeffs, sh_coeffs_rotated, rtol=0, atol=1e-6)
