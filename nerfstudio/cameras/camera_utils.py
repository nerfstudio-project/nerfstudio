# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Camera transformation helper code.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torchtyping import TensorType
from typing_extensions import Literal

_EPS = np.finfo(float).eps * 4.0


def unit_vector(data: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_from_matrix(matrix: ArrayLike, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0: ArrayLike, quat1: ArrayLike, fraction: float, spin: int = 0, shortestpath: bool = True
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_matrix(quaternion: ArrayLike) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_interpolated_poses(pose_a: ArrayLike, pose_b: ArrayLike, steps: int = 10) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        poseA: first pose
        poseB: second pose
        steps: number of steps the interpolated pose path should contain
    """

    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])

    ts = np.linspace(0, 1, steps)
    quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []
    for quat, tran in zip(quats, trans):
        pose = np.identity(4)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(pose[:3])
    return poses_ab


def get_interpolated_k(k_a, k_b, steps: int = 10) -> TensorType[3, 4]:
    """
    Returns interpolated path between two camera poses with specified number of steps.

    Args:
        KA: camera matrix 1
        KB: camera matrix 2
        steps: number of steps the interpolated pose path should contain
    """
    Ks = []
    ts = np.linspace(0, 1, steps)
    for t in ts:
        new_k = k_a * (1.0 - t) + k_b * t
        Ks.append(new_k)
    return Ks


def get_interpolated_poses_many(
    poses: TensorType["num_poses", 3, 4],
    Ks: TensorType["num_poses", 3, 3],
    steps_per_transition=10,
) -> Tuple[TensorType["num_poses", 3, 4], TensorType["num_poses", 3, 3]]:
    """Return interpolated poses for many camera poses.

    Args:
        poses: list of camera poses
        Ks: list of camera intrinsics
        steps_per_transition: number of steps per transition

    Returns:
        tuple of new poses and intrinsics
    """
    traj = []
    k_interp = []
    for idx in range(poses.shape[0] - 1):
        pose_a = poses[idx]
        pose_b = poses[idx + 1]
        poses_ab = get_interpolated_poses(pose_a, pose_b, steps=steps_per_transition)
        traj += poses_ab
        k_interp += get_interpolated_k(Ks[idx], Ks[idx + 1], steps=steps_per_transition)

    traj = np.stack(traj, axis=0)
    k_interp = np.stack(k_interp, axis=0)

    return torch.tensor(traj, dtype=torch.float32), torch.tensor(k_interp, dtype=torch.float32)


def normalize(x: torch.Tensor) -> TensorType[...]:
    """Returns a normalized vector."""
    return x / torch.linalg.norm(x)


def normalize_with_norm(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Args:
        x: tensor to normalize.
        dim: axis along which to normalize.

    Returns:
        Tuple of normalized tensor and corresponding norm.
    """

    norm = torch.maximum(torch.linalg.vector_norm(x, dim=dim, keepdims=True), torch.tensor([_EPS]).to(x))
    return x / norm, norm


def viewmatrix(lookat: torch.Tensor, up: torch.Tensor, pos: torch.Tensor) -> TensorType[...]:
    """Returns a camera transformation matrix.

    Args:
        lookat: The direction the camera is looking.
        up: The upward direction of the camera.
        pos: The position of the camera.

    Returns:
        A camera transformation matrix.
    """
    vec2 = normalize(lookat)
    vec1_avg = normalize(up)
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m


#@torch.jit.script
def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, p1, p2, k3, k4, k5, k6].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """
    assert distortion_params.shape[-1] == 8

    k1, k2, p1, p2, k3, k4, k5, k6 = torch.unbind(distortion_params, dim=-1)

    # let r(x, y) = x^2 + y^2;
    #     alpha(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    #     beta(x, y) = 1 + k4 * r(x, y) + k5 * r(x, y) ^2 + k6 * r(x, y)^3;
    #     d(x, y) = alpha(x, y) / beta(x, y);
    r = x * x + y * y
    alpha = 1.0 + r * (k1 + r * (k2 + r * k3))
    beta = 1.0 + r * (k4 + r * (k5 + r * k6))
    d = alpha / beta

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of alpha, beta over r.
    alpha_r = k1 + r * (2.0 * k2 + r * (3.0 * k3))
    beta_r = k4 + r * (2.0 * k5 + r * (3.0 * k6))

    # Compute derivative of d over [x, y]
    d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


#@torch.jit.script
def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
    resolution: torch.Tensor = torch.tensor([1e-3, 1e-3]),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters. Supports 0, 1, 2, 4, 8 parameters, in
            the order of [k1, k2, p1, p2, k3, k4, k5, k6].
        eps: The smallest determinant magnitude for a matrix to be considered invertible (for Newton's method).
        max_iterations: The maximum number of iterations to perform.
        resolution: The resolution [w, h] of the cameras

    Returns:
        The undistorted coordinates.
    """
    assert distortion_params.shape[-1] in [0, 1, 2, 4, 8]

    if distortion_params.shape[-1] == 0:
        return coords, torch.eye(2, device=coords.device), coords

    if distortion_params.shape[-1] < 8:
        distortion_params = F.pad(distortion_params, (0, 8 - distortion_params.shape[-1]), "constant", 0.0)
    assert distortion_params.shape[-1] == 8

    resolution = resolution.to(coords.device)
    # n_samples = coords.shape[0]
    # n_iters = 0

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]
    all_jacobian = torch.empty(x.shape + (2, 2), device=coords.device)
    all_residual = torch.empty_like(coords)

    next_upd = torch.arange(coords.shape[0], device=coords.device)

    for i in range(max_iterations):
        # n_iters += next_upd.shape[0]

        x_upd = x[next_upd]
        y_upd = y[next_upd]
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            # x=x, y=y,
            x=x_upd, y=y_upd,
            xd=coords[next_upd, 0], yd=coords[next_upd, 1],
            # xd=coords[:, 0], yd=coords[:, 1],
            distortion_params=distortion_params[next_upd]
        )

        max_x = torch.max(torch.abs(fx / resolution[next_upd, 0])).item()
        max_y = torch.max(torch.abs(fy / resolution[next_upd, 1])).item()
        print(f"iteration {i}, max residual {max_x}, {max_y}")

        converged = (fx < resolution[next_upd, 0] / 2) & (fy < resolution[next_upd, 1] / 2)

        not_converged = torch.argwhere(~converged).squeeze(-1)
        converged = torch.argwhere(converged).squeeze(-1)

        # print(f"{converged.shape[0]} points converged")

        denominator = fx_x * fy_y - fx_y * fy_x
        invertible = torch.abs(denominator) > eps
        numerator = torch.stack([fy_y, -fx_y, -fy_x, fx_x], dim=-1)
        j_inv = (torch.where(invertible, 1 / denominator, 0).reshape(-1, 1) * numerator).reshape(-1, 2, 2)
        upd_conv = next_upd[converged]
        all_jacobian[upd_conv] = j_inv[converged]
        residual = torch.stack((fx, fy), dim=-1)
        all_residual[upd_conv] = residual[converged]

        next_upd = next_upd[not_converged]
        if next_upd.numel() == 0:
            break

        j_inv = j_inv[not_converged]

        residual = residual[not_converged].reshape(-1, 2, 1)
        step = (j_inv @ residual).squeeze().to(torch.float32)

        # careful: index_add_ (with underscore) is in-place, index_add (no underscore) is not in-place
        x = x.index_add(dim=0, index=next_upd, source=step[:, 0], alpha=-1)
        y = y.index_add(dim=0, index=next_upd, source=step[:, 1], alpha=-1)

    # print(f"average number of newton iterations per sample: {n_iters / n_samples}")

    undistort = torch.stack([x, y], dim=-1)

    return undistort, all_jacobian, coords + all_residual


@torch.jit.script
def _compute_residual_and_jacobian_old(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, p1, p2, k3, k4, k5, k6].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """
    assert distortion_params.shape[-1] == 8

    k1, k2, p1, p2, k3, k4, k5, k6 = torch.unbind(distortion_params, dim=-1)

    # let r(x, y) = x^2 + y^2;
    #     alpha(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    #     beta(x, y) = 1 + k4 * r(x, y) + k5 * r(x, y) ^2 + k6 * r(x, y)^3;
    #     d(x, y) = alpha(x, y) / beta(x, y);
    r = x * x + y * y
    alpha = 1.0 + r * (k1 + r * (k2 + r * k3))
    beta = 1.0 + r * (k4 + r * (k5 + r * k6))
    d = alpha / beta

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of alpha, beta over r.
    alpha_r = k1 + r * (2.0 * k2 + r * (3.0 * k3))
    beta_r = k4 + r * (2.0 * k5 + r * (3.0 * k6))

    # Compute derivative of d over [x, y]
    d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


@torch.jit.script
def radial_and_tangential_undistort_old(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    """Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters. Supports 0, 1, 2, 4, 8 parameters, in
            the order of [k1, k2, p1, p2, k3, k4, k5, k6].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """
    assert distortion_params.shape[-1] in [0, 1, 2, 4, 8]

    if distortion_params.shape[-1] == 0:
        return coords

    if distortion_params.shape[-1] < 8:
        distortion_params = F.pad(distortion_params, (0, 8 - distortion_params.shape[-1]), "constant", 0.0)
    assert distortion_params.shape[-1] == 8

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(torch.abs(denominator) > eps, x_numerator / denominator, torch.zeros_like(denominator))
        step_y = torch.where(torch.abs(denominator) > eps, y_numerator / denominator, torch.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


@torch.jit.script
def _compute_residual_and_jacobian_fisheye(
    theta: torch.Tensor,
    thetad: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        theta: The updated angles.
        thetad: The distorted angles.
        distortion_params: The distortion parameters [k1, k2, k3, k4].

    Returns:
        The residuals (ftheta) and derivatives (ftheta_theta).
    """
    assert distortion_params.shape[-1] == 4

    k1, k2, k3, k4 = torch.unbind(distortion_params, dim=-1)

    # let d(theta) = 1 + k1 * theta^2 + k2 * theta^4 + k3 * theta^6 + k4 * theta^8
    # r(theta) = theta^2
    r = theta * theta
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # thetad = theta * d(theta)
    #
    # Let's define
    #
    # f(theta) = theta * d(theta) - thetad;
    #
    # We are looking for a solution that satisfies
    # f(theta) = 0;
    f = theta * d - thetad

    # Compute derivative of f over theta.
    f_theta = 1 + r * (3 * k1 + r * (5 * k2 + r * (7 * k3 + r * 9 * k4)))

    return f, f_theta


@torch.jit.script
def fisheye_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
    resolution: torch.Tensor = torch.tensor([1e-3, 1e-3]),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes undistorted coords given opencv distortion parameters. Based on OpenCV fisheye camera model.

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters. Supports 0, 1, 2, 4 parameters, in
            the order of [k1, k2, k3, k4].
        eps: The smallest derivative magnitude considered to be nonzero (for Newton's method).
        max_iterations: The maximum number of iterations to perform.
        resolution: The resolution [w, h] of the cameras

    Returns:
        The undistorted coordinates.
    """
    assert distortion_params.shape[-1] in [0, 1, 2, 4]

    if distortion_params.shape[-1] == 0:
        return coords, torch.tensor(1, device=coords.device), coords

    if distortion_params.shape[-1] < 4:
        distortion_params = F.pad(distortion_params, (0, 4 - distortion_params.shape[-1]), "constant", 0.0)
    assert distortion_params.shape[-1] == 4

    resolution, _ = torch.min(resolution.to(coords.device), dim=-1)
    # n_samples = coords.shape[0]
    # n_iters = 0

    # OpenCV uses an equidistant projection for its fisheye camera model
    # r = theta
    r_d = torch.linalg.norm(coords, dim=-1)

    # Initialize from the distorted point.
    theta = torch.clone(r_d)
    all_derivative = torch.empty_like(theta)
    all_residual = torch.empty_like(theta)

    next_upd = torch.arange(theta.shape[0], device=theta.device)

    for i in range(max_iterations):
        # n_iters += next_upd.shape[0]

        theta_upd = theta[next_upd]
        f, dtheta = _compute_residual_and_jacobian_fisheye(
            theta=theta_upd,
            thetad=r_d[next_upd],
            distortion_params=distortion_params[next_upd]
        )

        # max_x = torch.max(torch.abs(fx * resolution[..., 0])).item()
        # max_y = torch.max(torch.abs(fy * resolution[..., 1])).item()
        # print(f"iteration {i}, max residual {max_x}, {max_y}")

        converged = f < resolution[next_upd]

        not_converged = torch.argwhere(~converged).squeeze()
        converged = torch.argwhere(converged).squeeze()

        # print(f"{converged.shape[0]} points converged")

        upd_conv = next_upd[converged]
        all_derivative[upd_conv] = dtheta[converged]
        all_residual[upd_conv] = f[converged]

        next_upd = next_upd[not_converged]
        if next_upd.numel() == 0:
            break

        dtheta = dtheta[not_converged]

        f = f[not_converged].reshape(-1, 1)
        step = torch.where(dtheta > eps, (f / dtheta), 0)

        # careful: index_add_ (with underscore) is in-place, index_add (no underscore) is not in-place
        theta = theta.index_add(dim=0, index=next_upd, source=step[:, 0], alpha=-1)

    # print(f"average number of newton iterations per sample: {n_iters / n_samples}")
    return (theta / r_d).unsqueeze(-1) * coords, all_derivative, coords * (1 + all_residual / r_d).unsqueeze(-1)

def rotation_matrix(a: TensorType[3], b: TensorType[3]) -> TensorType[3, 3]:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def focus_of_attention(poses: TensorType["num_poses":..., 4, 4], initial_focus: TensorType[3]) -> TensorType[3]:
    """Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

     Args:
        poses: The poses to orient.
        initial_focus: The 3D point views to decide which cameras are initially activated.

    Returns:
        The 3D position of the focus of attention.
    """
    # References to the same method in third-party code:
    # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    # initial value for testing if the focus_pt is in front or behind
    focus_pt = initial_focus
    # Prune cameras which have the current have the focus_pt behind them.
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    # We need at least two active cameras, else fallback on the previous solution.
    # This may be the "poses" solution if no cameras are active on first iteration, e.g.
    # they are in an outward-looking configuration.
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
        if active.all():
            # the set of active cameras did not change, so we're done.
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses: TensorType["num_poses":..., 4, 4],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[TensorType["num_poses":..., 3, 4], TensorType[3, 4]]:
    """Orients and centers the poses. We provide two methods for orientation: pca and up.

    pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:
    poses: The poses are centered around the origin.
    focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform
