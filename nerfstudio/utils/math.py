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

""" Math Helper Functions """

from dataclasses import dataclass
from typing import Literal, Tuple

import math
import torch
import numpy as np
from jaxtyping import Bool, Float
from torch import Tensor

from nerfstudio.utils.misc import torch_compile


def components_from_spherical_harmonics(
    levels: int, directions: Float[Tensor, "*batch 3"]
) -> Float[Tensor, "*batch components"]:
    """
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = levels**2
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
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
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
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


@dataclass
class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """

    mean: Float[Tensor, "*batch dim"]
    cov: Float[Tensor, "*batch dim dim"]


def compute_3d_gaussian(
    directions: Float[Tensor, "*batch 3"],
    means: Float[Tensor, "*batch 3"],
    dir_variance: Float[Tensor, "*batch 1"],
    radius_variance: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Compute gaussian along ray.

    Args:
        directions: Axis of Gaussian.
        means: Mean of Gaussian.
        dir_variance: Variance along direction axis.
        radius_variance: Variance tangent to direction axis.

    Returns:
        Gaussians: Oriented 3D gaussian.
    """

    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = torch.eye(directions.shape[-1], device=directions.device)
    dir_mag_sq = torch.clamp(torch.sum(directions**2, dim=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    cov = dir_cov_diag + radius_cov_diag
    return Gaussians(mean=means, cov=cov)


def cylinder_to_gaussian(
    origins: Float[Tensor, "*batch 3"],
    directions: Float[Tensor, "*batch 3"],
    starts: Float[Tensor, "*batch 1"],
    ends: Float[Tensor, "*batch 1"],
    radius: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Approximates cylinders with a Gaussian distributions.

    Args:
        origins: Origins of cylinders.
        directions: Direction (axis) of cylinders.
        starts: Start of cylinders.
        ends: End of cylinders.
        radius: Radii of cylinders.

    Returns:
        Gaussians: Approximation of cylinders
    """
    means = origins + directions * ((starts + ends) / 2.0)
    dir_variance = (ends - starts) ** 2 / 12
    radius_variance = radius**2 / 4.0
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def conical_frustum_to_gaussian(
    origins: Float[Tensor, "*batch 3"],
    directions: Float[Tensor, "*batch 3"],
    starts: Float[Tensor, "*batch 1"],
    ends: Float[Tensor, "*batch 1"],
    radius: Float[Tensor, "*batch 1"],
) -> Gaussians:
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw**2.0) / (3.0 * mu**2.0 + hw**2.0))
    dir_variance = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
    radius_variance = radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def multisampled_frustum_to_gaussian(
    origins: Float[Tensor, "*batch num_samples 3"],
    directions: Float[Tensor, "*batch num_samples 3"],
    starts: Float[Tensor, "*batch num_samples 1"],
    ends: Float[Tensor, "*batch num_samples 1"],
    radius: Float[Tensor, "*batch num_samples 1"],
    rand: bool = True,
    cov_scale: float = 0.5,
    eps: float = 1e-10,
) -> Gaussians:
    """Approximates frustums with a Gaussian distributions via multisampling.
    Proposed in ZipNeRF https://arxiv.org/pdf/2304.06706.pdf

    Taken from https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/b1cb42943d244301a013bd53f9cb964f576b0af4/internal/render.py#L92

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of frustums.
        ends: End of frustums.
        radius: Radii of cone a distance of 1 from the origin.
        rand: Whether should add noise to points or not.
        cov_scale: Covariance scale parameter.
        eps: Small number.

    Returns:
        Gaussians: Approximation of frustums via multisampling
    """

    # middle points
    t_m = (starts + ends) / 2.0
    # half of the width
    t_d = (ends - starts) / 2.0

    # prepare 6-point hexagonal pattern for each sample
    j = torch.arange(6, device=starts.device, dtype=starts.dtype)
    t = starts + t_d / (t_d**2 + 3 * t_m**2) * (
        ends**2 + 2 * t_m**2 + 3 / 7**0.5 * (2 * j / 5 - 1) * ((t_d**2 - t_m**2) ** 2 + 4 * t_m**4).sqrt()
    )  # [..., num_samples, 6]

    deg = torch.pi / 3 * starts.new_tensor([0, 2, 4, 3, 5, 1]).broadcast_to(t.shape)
    if rand:
        # randomly rotate and flip
        mask = torch.rand_like(starts) > 0.5  # [..., num_samples, 1]
        deg = deg + 2 * torch.pi * torch.rand_like(starts)
        deg = torch.where(mask, deg, 5 * torch.pi / 3.0 - deg)
    else:
        # rotate 30 degree and flip every other pattern
        mask = (
            (
                torch.arange(
                    end=starts.shape[-2],
                    device=starts.device,
                    dtype=starts.dtype,
                )
                % 2
                == 0
            )
            .unsqueeze(-1)
            .broadcast_to(starts.shape)
        )  # [..., num_samples, 6]
        deg = torch.where(mask, deg, deg + torch.pi / 6.0)
        deg = torch.where(mask, deg, 5 * torch.pi / 3.0 - deg)

    means = torch.stack(
        [
            radius * t * torch.cos(deg) / 2**0.5,
            radius * t * torch.sin(deg) / 2**0.5,
            t,
        ],
        dim=-1,
    )  # [..., "num_samples", 6, 3]
    stds = cov_scale * radius * t / 2**0.5  # [..., "num_samples", 6]

    # extend stds as diagonal
    # stds = stds.unsqueeze(-1).broadcast_to(*stds.shape, 3).diag_embed() # [..., "num_samples", 6, 3, 3]

    # two basis in parallel to the image plane
    rand_vec = torch.rand(
        list(directions.shape[:-2]) + [1, 3],
        device=directions.device,
        dtype=directions.dtype,
    )  # [..., 1, 3]
    ortho1 = torch.nn.functional.normalize(
        torch.cross(directions, rand_vec, dim=-1), dim=-1, eps=eps
    )  # [..., num_samples, 3]
    ortho2 = torch.nn.functional.normalize(
        torch.cross(directions, ortho1, dim=-1), dim=-1, eps=eps
    )  # [..., num_samples, 3]

    # just use directions to be the third vector of the orthonormal basis,
    # while the cross section of cone is parallel to the image plane
    basis_matrix = torch.stack([ortho1, ortho2, directions], dim=-1)
    means = torch.matmul(means, basis_matrix.transpose(-1, -2))  # [..., "num_samples", 6, 3]
    means = means + origins[..., None, :]

    return Gaussians(mean=means, cov=stds)


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """

    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def intersect_aabb(
    origins: torch.Tensor,
    directions: torch.Tensor,
    aabb: torch.Tensor,
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of ray intersection with AABB box

    Args:
        origins: [N,3] tensor of 3d positions
        directions: [N,3] tensor of normalized directions
        aabb: [6] array of aabb box in the form of [x_min, y_min, z_min, x_max, y_max, z_max]
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection

    Returns:
        t_min, t_max - two tensors of shapes N representing distance of intersection from the origin.
    """

    tx_min = (aabb[:3] - origins) / directions
    tx_max = (aabb[3:] - origins) / directions

    t_min = torch.stack((tx_min, tx_max)).amin(dim=0)
    t_max = torch.stack((tx_min, tx_max)).amax(dim=0)

    t_min = t_min.amax(dim=-1)
    t_max = t_max.amin(dim=-1)

    t_min = torch.clamp(t_min, min=0, max=max_bound)
    t_max = torch.clamp(t_max, min=0, max=max_bound)

    cond = t_max <= t_min
    t_min = torch.where(cond, invalid_value, t_min)
    t_max = torch.where(cond, invalid_value, t_max)

    return t_min, t_max


def safe_normalize(
    vectors: Float[Tensor, "*batch_dim N"],
    eps: float = 1e-10,
) -> Float[Tensor, "*batch_dim N"]:
    """Normalizes vectors.

    Args:
        vectors: Vectors to normalize.
        eps: Epsilon value to avoid division by zero.

    Returns:
        Normalized vectors.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + eps)


def masked_reduction(
    input_tensor: Float[Tensor, "1 32 mult"],
    mask: Bool[Tensor, "1 32 mult"],
    reduction_type: Literal["image", "batch"],
) -> Tensor:
    """
    Whether to consolidate the input_tensor across the batch or across the image
    Args:
        input_tensor: input tensor
        mask: mask tensor
        reduction_type: either "batch" or "image"
    Returns:
        input_tensor: reduced input_tensor
    """
    if reduction_type == "batch":
        # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
        divisor = torch.sum(mask)
        if divisor == 0:
            return torch.tensor(0, device=input_tensor.device)
        input_tensor = torch.sum(input_tensor) / divisor
    elif reduction_type == "image":
        # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
        valid = mask.nonzero()

        input_tensor[valid] = input_tensor[valid] / mask[valid]
        input_tensor = torch.mean(input_tensor)
    return input_tensor


def normalized_depth_scale_and_shift(
    prediction: Float[Tensor, "1 32 mult"], target: Float[Tensor, "1 32 mult"], mask: Bool[Tensor, "1 32 mult"]
):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift


def power_fn(x: torch.Tensor, lam: float = -1.5, max_bound: float = 1e10) -> torch.Tensor:
    """Power transformation function from Eq. 4 in ZipNeRF paper."""

    if lam == 1:
        return x
    if lam == 0:
        return torch.log1p(x)
    # infinity case
    if lam > max_bound:
        return torch.expm1(x)
    # -infinity case
    if lam < -max_bound:
        return -torch.expm1(-x)

    lam_1 = abs(lam - 1)
    return (lam_1 / lam) * ((x / lam_1 + 1) ** lam - 1)


def inv_power_fn(
    x: torch.Tensor,
    lam: float = -1.5,
    eps: float = 1e-10,
    max_bound: float = 1e10,
) -> torch.Tensor:
    """Inverse power transformation function from Eq. 4 in ZipNeRF paper."""

    if lam == 1:
        return x
    if lam == 0:
        return torch.expm1(x)
    # infinity case
    if lam > max_bound:
        return torch.log1p(x)
    # -infinity case
    if lam < -max_bound:
        return -torch.log(1 - x)

    lam_1 = abs(lam - 1)
    return ((x * lam / lam_1 + 1).clamp_min(eps) ** (1 / lam) - 1) * lam_1


@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def erf_approx(x: torch.Tensor) -> torch.Tensor:
    """Error function approximation proposed in ZipNeRF paper (Eq. 11)."""
    return torch.sign(x) * torch.sqrt(1 - torch.exp(-4 / torch.pi * x**2))


def div_round_up(val: int, divisor: int) -> int:
    return (val + divisor - 1) // divisor


def grid_scale(level: int, log2_per_level_scale: float, base_resolution: int) -> float:
    # The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
    # than the number of cells. This is slightly different from the notation in the paper,
    # but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
    return np.exp2(level * log2_per_level_scale) * base_resolution - 1.0


def grid_resolution(scale: float) -> int:
    return math.ceil(scale) + 1


def powi(base: int, exponent: int) -> int:
    result: int = 1
    for _ in range(exponent):
        result *= base
    return result


def next_multiple(val: int, divisor: int) -> int:
    return div_round_up(val, divisor) * divisor
