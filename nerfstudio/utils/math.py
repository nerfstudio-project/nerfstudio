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

import itertools
import math
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from nerfstudio.data.scene_box import OrientedBox


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


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


# @torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
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


def intersect_obb(
    origins: torch.Tensor,
    directions: torch.Tensor,
    obb: OrientedBox,
    max_bound: float = 1e10,
    invalid_value: float = 1e10,
):
    """
    Ray intersection with an oriented bounding box (OBB)

    Args:
        origins: [N,3] tensor of 3d positions
        directions: [N,3] tensor of normalized directions
        R: [3,3] rotation matrix
        T: [3] translation vector
        S: [3] extents of the bounding box
        max_bound: Maximum value of t_max
        invalid_value: Value to return in case of no intersection
    """
    # Transform ray to OBB space
    R, T, S = obb.R, obb.T, obb.S.to(origins.device)
    H = torch.eye(4, device=origins.device, dtype=origins.dtype)
    H[:3, :3] = R
    H[:3, 3] = T
    H_world2bbox = torch.inverse(H)
    origins = torch.cat((origins, torch.ones_like(origins[..., :1])), dim=-1)
    origins = torch.matmul(H_world2bbox, origins.T).T[..., :3]
    directions = torch.matmul(H_world2bbox[:3, :3], directions.T).T

    # Compute intersection with axis-aligned bounding box with min as -S and max as +S
    aabb = torch.concat((-S / 2, S / 2))
    t_min, t_max = intersect_aabb(origins, directions, aabb, max_bound=max_bound, invalid_value=invalid_value)

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


def columnwise_squared_l2_distance(
    x: Float[Tensor, "*M N"],
    y: Float[Tensor, "*M N"],
) -> Float[Tensor, "N N"]:
    """Compute the squared Euclidean distance between all pairs of columns.
    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        x: tensor of floats, with shape [M, N].
        y: tensor of floats, with shape [M, N].
    Returns:
        sq_dist: tensor of floats, with shape [N, N].
    """
    # Use the fact that ||x - y||^2 == ||x||^2 + ||y||^2 - 2 x^T y.
    sq_norm_x = torch.sum(x**2, 0)
    sq_norm_y = torch.sum(y**2, 0)
    sq_dist = sq_norm_x[:, None] + sq_norm_y[None, :] - 2 * x.T @ y
    return sq_dist


def _compute_tesselation_weights(v: int) -> Tensor:
    """Tesselate the vertices of a triangle by a factor of `v`.
    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        v: int, the factor of the tesselation (v==1 is a no-op to the triangle).

    Returns:
        weights: tesselated weights.
    """
    if v < 1:
        raise ValueError(f"v {v} must be >= 1")
    int_weights = []
    for i in range(v + 1):
        for j in range(v + 1 - i):
            int_weights.append((i, j, v - (i + j)))
    int_weights = torch.FloatTensor(int_weights)
    weights = int_weights / v  # Barycentric weights.
    return weights


def _tesselate_geodesic(
    vertices: Float[Tensor, "N 3"], faces: Float[Tensor, "M 3"], v: int, eps: float = 1e-4
) -> Tensor:
    """Tesselate the vertices of a geodesic polyhedron.

    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        vertices: tensor of floats, the vertex coordinates of the geodesic.
        faces: tensor of ints, the indices of the vertices of base_verts that
            constitute eachface of the polyhedra.
        v: int, the factor of the tesselation (v==1 is a no-op).
        eps: float, a small value used to determine if two vertices are the same.

    Returns:
        verts: a tensor of floats, the coordinates of the tesselated vertices.
    """
    tri_weights = _compute_tesselation_weights(v)

    verts = []
    for face in faces:
        new_verts = torch.matmul(tri_weights, vertices[face, :])
        new_verts /= torch.sqrt(torch.sum(new_verts**2, 1, keepdim=True))
        verts.append(new_verts)
    verts = torch.concatenate(verts, 0)

    sq_dist = columnwise_squared_l2_distance(verts.T, verts.T)
    assignment = torch.tensor([torch.min(torch.argwhere(d <= eps)) for d in sq_dist])
    unique = torch.unique(assignment)
    verts = verts[unique, :]
    return verts


def generate_polyhedron_basis(
    basis_shape: Literal["icosahedron", "octahedron"],
    angular_tesselation: int,
    remove_symmetries: bool = True,
    eps: float = 1e-4,
) -> Tensor:
    """Generates a 3D basis by tesselating a geometric polyhedron.
    Basis is used to construct Fourier features for positional encoding.
    See Mip-Nerf360 paper: https://arxiv.org/abs/2111.12077
    Adapted from https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/geopoly.py

    Args:
        base_shape: string, the name of the starting polyhedron, must be either
            'icosahedron' or 'octahedron'.
        angular_tesselation: int, the number of times to tesselate the polyhedron,
            must be >= 1 (a value of 1 is a no-op to the polyhedron).
        remove_symmetries: bool, if True then remove the symmetric basis columns,
            which is usually a good idea because otherwise projections onto the basis
            will have redundant negative copies of each other.
        eps: float, a small number used to determine symmetries.

    Returns:
        basis: a matrix with shape [3, n].
    """
    if basis_shape == "icosahedron":
        a = (math.sqrt(5) + 1) / 2
        verts = torch.FloatTensor(
            [
                (-1, 0, a),
                (1, 0, a),
                (-1, 0, -a),
                (1, 0, -a),
                (0, a, 1),
                (0, a, -1),
                (0, -a, 1),
                (0, -a, -1),
                (a, 1, 0),
                (-a, 1, 0),
                (a, -1, 0),
                (-a, -1, 0),
            ]
        ) / math.sqrt(a + 2)
        faces = torch.tensor(
            [
                (0, 4, 1),
                (0, 9, 4),
                (9, 5, 4),
                (4, 5, 8),
                (4, 8, 1),
                (8, 10, 1),
                (8, 3, 10),
                (5, 3, 8),
                (5, 2, 3),
                (2, 7, 3),
                (7, 10, 3),
                (7, 6, 10),
                (7, 11, 6),
                (11, 0, 6),
                (0, 1, 6),
                (6, 1, 10),
                (9, 0, 11),
                (9, 11, 2),
                (9, 2, 5),
                (7, 2, 11),
            ]
        )
        verts = _tesselate_geodesic(verts, faces, angular_tesselation)
    elif basis_shape == "octahedron":
        verts = torch.FloatTensor([(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)])
        corners = torch.FloatTensor(list(itertools.product([-1, 1], repeat=3)))
        pairs = torch.argwhere(columnwise_squared_l2_distance(corners.T, verts.T) == 2)
        faces, _ = torch.sort(torch.reshape(pairs[:, 1], [3, -1]).T, 1)
        verts = _tesselate_geodesic(verts, faces, angular_tesselation)

    if remove_symmetries:
        # Remove elements of `verts` that are reflections of each other.
        match = columnwise_squared_l2_distance(verts.T, -verts.T) < eps
        verts = verts[torch.any(torch.triu(match), 1), :]

    basis = verts.flip(-1)
    return basis
