# Licensed under the Apache License, Version 2.0.
"""Independent geometry invariants for PCA camera normalization."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[2] / "nerfstudio" / "cameras" / "camera_utils.py"
spec = importlib.util.spec_from_file_location("nerfstudio_camera_utils", path)
camera_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_utils)


def _scene(dtype, up_sign):
    # A nonsymmetric rotation avoids accidentally identifying Q with Q.T.
    az = torch.tensor(0.37, dtype=dtype)
    ay = torch.tensor(-0.61, dtype=dtype)
    cz, sz, cy, sy = az.cos(), az.sin(), ay.cos(), ay.sin()
    rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=dtype)
    world_rotation = rz @ ry
    local_centers = torch.cat(
        [
            torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype)),
            -torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype)),
        ]
    )
    offset = torch.tensor([4.0, -3.0, 2.0], dtype=dtype)
    centers = local_centers @ world_rotation.T + offset
    # Local up is the unique smallest principal axis. Both signs cover the
    # orientation correction regardless of eigenvector sign conventions.
    local_camera_rotation = torch.tensor([[0.0, up_sign, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -up_sign]], dtype=dtype)
    poses = torch.eye(4, dtype=dtype).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = world_rotation @ local_camera_rotation
    poses[:, :3, 3] = centers
    return poses


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("up_sign", [-1.0, 1.0])
def test_pca_aligns_principal_axes_in_variance_order(dtype, up_sign):
    poses = _scene(dtype, up_sign)
    oriented, _ = camera_utils.auto_orient_and_center_poses(poses, method="pca")
    centers = oriented[:, :3, 3]
    covariance = centers.T @ centers
    # Six analytic centers give total variances 2*3^2, 2*2^2, 2*1^2.
    expected = torch.diag(torch.tensor([18.0, 8.0, 2.0], dtype=dtype))
    torch.testing.assert_close(covariance, expected, atol=1e-5 if dtype == torch.float32 else 1e-12, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("up_sign", [-1.0, 1.0])
def test_pca_applies_one_proper_transform_to_all_cameras(dtype, up_sign):
    poses = _scene(dtype, up_sign)
    before = poses.clone()
    oriented, transform = camera_utils.auto_orient_and_center_poses(poses, method="pca")
    torch.testing.assert_close(oriented, transform @ poses)
    torch.testing.assert_close(torch.linalg.det(oriented[:, :3, :3]), torch.ones(len(poses), dtype=dtype))
    torch.testing.assert_close(torch.linalg.det(transform[:, :3]), torch.tensor(1.0, dtype=dtype))
    torch.testing.assert_close(
        oriented[:, :3, 3].mean(0), torch.zeros(3, dtype=dtype), atol=1e-6 if dtype == torch.float32 else 1e-12, rtol=0
    )
    assert oriented[:, 2, 1].mean() >= 0
    torch.testing.assert_close(poses, before, rtol=0, atol=0)
    assert transform.dtype == poses.dtype and oriented.dtype == poses.dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("up_sign", [-1.0, 1.0])
def test_pca_preserves_camera_coordinates_and_projections(dtype, up_sign):
    poses = _scene(dtype, up_sign)
    world_points = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, -2.0], [-4.0, 2.0, 1.0]], dtype=dtype)
    oriented, transform = camera_utils.auto_orient_and_center_poses(poses, method="pca")
    transformed_points = world_points @ transform[:, :3].T + transform[:, 3]
    before = torch.einsum("nji,npj->npi", poses[:, :3, :3], world_points[None] - poses[:, None, :3, 3])
    after = torch.einsum("nji,npj->npi", oriented[:, :3, :3], transformed_points[None] - oriented[:, None, :3, 3])
    torch.testing.assert_close(after, before, atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=2e-6)
    torch.testing.assert_close(
        after[..., :2] / after[..., 2:],
        before[..., :2] / before[..., 2:],
        atol=1e-5 if dtype == torch.float32 else 1e-12,
        rtol=2e-6,
    )
