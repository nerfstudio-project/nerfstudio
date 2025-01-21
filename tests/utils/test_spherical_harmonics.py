from typing import Literal

import numpy as np
import pytest
import torch
from gsplat.cuda._torch_impl import (
    _eval_sh_bases_fast as gsplat_eval_sh_bases,
    _spherical_harmonics as gsplat_spherical_harmonics,
)
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
def test_spherical_harmonics_rotation_nerfacto(sh_degree):
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
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs, component_convention="+y,+z,+x")
    dirs_rotated = (rot_matrix @ dirs.T).T
    y_lm_rotated = components_from_spherical_harmonics(sh_degree, dirs_rotated)
    color_rotated = (sh_coeffs_rotated * y_lm_rotated[..., None, :]).sum(dim=-1)

    torch.testing.assert_close(color_original, color_rotated, rtol=0, atol=1e-5)


@pytest.mark.parametrize("sh_degree", list(range(0, 4)))
def test_spherical_harmonics_rotation_splatfacto(sh_degree):
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
    assert dirs.shape == (N, 3)

    rot_matrix = torch.tensor(ScR.random().as_matrix(), dtype=torch.float32)
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs, component_convention="-y,+z,-x")
    dirs_rotated = (rot_matrix @ dirs.T).T
    assert dirs_rotated.shape == (N, 3)

    torch.testing.assert_close(
        gsplat_spherical_harmonics(sh_degree, coeffs=sh_coeffs.swapaxes(-1, -2), dirs=dirs),
        gsplat_spherical_harmonics(sh_degree, coeffs=sh_coeffs_rotated.swapaxes(-1, -2), dirs=dirs_rotated),
    )


@pytest.mark.parametrize("sh_degree", list(range(0, 4)))
@pytest.mark.parametrize("component_convention", ["+y,+z,+x", "-y,+z,-x"])
def test_spherical_harmonics_rotation_properties(sh_degree: int, component_convention: Literal["+y,+z,+x", "-y,+z,-x"]):
    """Test properties of the SH rotation"""
    torch.manual_seed(0)
    np.random.seed(0)

    N = 1000
    num_coeffs = (sh_degree + 1) ** 2
    sh_coeffs = torch.rand(N, 3, num_coeffs)
    rot_matrix = torch.tensor(ScR.random().as_matrix(), dtype=torch.float32)
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs, component_convention)

    # Norm preserving
    norm_original = torch.norm(sh_coeffs, dim=-1)
    norm_rotated = torch.norm(sh_coeffs_rotated, dim=-1)
    torch.testing.assert_close(norm_original, norm_rotated, rtol=0, atol=1e-5)

    # 0th degree coeffs are invariant to rotation
    torch.testing.assert_close(sh_coeffs[..., 0], sh_coeffs_rotated[..., 0], rtol=0, atol=1e-6)

    # Identity rotation
    rot_matrix = torch.eye(3)
    sh_coeffs_rotated = rotate_spherical_harmonics(rot_matrix, sh_coeffs, ordering)
    torch.testing.assert_close(sh_coeffs, sh_coeffs_rotated, rtol=0, atol=1e-6)
