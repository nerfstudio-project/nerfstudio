# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphecal Harmonics utils."""

import math
from typing import Literal

import torch
from e3nn.o3 import Irreps
from jaxtyping import Float
from torch import Tensor
from typing_extensions import assert_never

MAX_SH_DEGREE = 4


def components_from_spherical_harmonics(
    degree: int, directions: Float[Tensor, "*batch 3"]
) -> Float[Tensor, "*batch components"]:
    """
    Returns value for each component of spherical harmonics.

    Args:
        degree: Number of spherical harmonic degrees to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = num_sh_bases(degree)
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 0 <= degree <= MAX_SH_DEGREE, f"SH degree must be in [0, {MAX_SH_DEGREE}], got {degree}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if degree > 0:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if degree > 1:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if degree > 2:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if degree > 3:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components


def num_sh_bases(degree: int) -> int:
    """
    Returns the number of spherical harmonic bases for a given degree.
    """
    assert degree <= MAX_SH_DEGREE, f"We don't support degree greater than {MAX_SH_DEGREE}."
    return (degree + 1) ** 2


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def rotate_spherical_harmonics(
    rotation_matrix: Float[Tensor, "3 3"],
    coeffs: Float[Tensor, "*batch dim_sh"],
    component_convention: Literal["-y,+z,-x", "+y,+z,+x"],
) -> Float[Tensor, "*batch dim_sh"]:
    """Rotates real spherical harmonic coefficients using a given 3x3 rotation matrix.

    Args:
        rotation_matrix : A 3x3 rotation matrix.
        coeffs : SH coefficients
        component_convention: Component convention for spherical harmonics.
            Nerfstudio (nerfacto) uses +y,+z,+x, while gsplat (splatfacto) uses
            -y,+z,-x.

    Returns:
        The rotated SH coefficients
    """
    dim_sh = coeffs.shape[-1]
    assert math.isqrt(dim_sh) ** 2 == dim_sh, "dim_sh must be a perfect square (l+1)^2"
    sh_degree = int(math.sqrt(dim_sh)) - 1

    # e3nn uses the xyz ordering instead of the standard yzx used in ns, equivalent to a change of basis
    if component_convention == "+y,+z,+x":
        R_xyz_from_yzx = torch.tensor(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        )
        rotation_matrix = (R_xyz_from_yzx.T @ rotation_matrix @ R_xyz_from_yzx).cpu()
    elif component_convention == "-y,+z,-x":
        R_xyz_from_negyznegx = torch.tensor(
            [
                [0, 0, -1],
                [-1, 0, 0],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        )
        rotation_matrix = (R_xyz_from_negyznegx.T @ rotation_matrix @ R_xyz_from_negyznegx).cpu()
    else:
        assert_never(component_convention)

    irreps = Irreps(" + ".join([f"{i}e" for i in range(sh_degree + 1)]))  # Even parity spherical harmonics of degree l
    D_matrix = irreps.D_from_matrix(rotation_matrix).to(coeffs.device)  # Construct Wigner D-matrix

    # Multiply last dimension of coeffs (..., dim_sh) with the Wigner D-matrix (dim_sh, dim_sh)
    rotated_coeffs = coeffs @ D_matrix.T
    return rotated_coeffs
