# Licensed under the Apache License, Version 2.0.
"""SO(3) invariants for rotations between camera direction vectors."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[2] / "nerfstudio" / "cameras" / "camera_utils.py"
spec = importlib.util.spec_from_file_location("nerfstudio_camera_utils", path)
camera_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_utils)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-8, 5e-7, 1e-5, 0.7, 2.3, torch.pi - 1e-5, torch.pi])
def test_rotation_maps_directions_and_preserves_so3(dtype, angle):
    a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    if angle == torch.pi:
        b = -a
    else:
        theta = torch.tensor(angle, dtype=dtype)
        b = torch.stack((theta.cos(), theta.sin(), theta.new_zeros(())))
    before_a, before_b = a.clone(), b.clone()
    rotation = camera_utils.rotation_matrix_between(a, b)
    assert rotation.dtype == dtype
    assert rotation.device == a.device
    # A residual threshold alone could let every tiny-angle case pass.
    # Their relative vector error must also remain below 1 percent.
    tolerance = max(2e-12, abs(float(b[1])) * 0.01)
    torch.testing.assert_close(rotation @ a, b, rtol=1e-6, atol=tolerance)
    identity = torch.eye(3, dtype=dtype)
    torch.testing.assert_close(rotation.T @ rotation, identity)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0, dtype=dtype))
    # Minimal-angle rotations satisfy trace(R) = 1 + 2 * dot(a, b).
    torch.testing.assert_close(rotation.trace(), 1 + 2 * torch.dot(a, b))
    torch.testing.assert_close(a, before_a, rtol=0, atol=0)
    torch.testing.assert_close(b, before_b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonaxis_parallel_and_antiparallel_vectors(dtype):
    a = torch.tensor([1.0, 2.0, -3.0], dtype=dtype)
    for factor in [-1.0, 1.0]:
        b = factor * a
        rotation = camera_utils.rotation_matrix_between(a, b)
        assert rotation.dtype == dtype
        torch.testing.assert_close(rotation @ a, b)
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(3, dtype=dtype))
        torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_is_invariant_to_direction_magnitudes(dtype):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    b = torch.tensor([-2.0, 1.0, 4.0], dtype=dtype)
    reference = camera_utils.rotation_matrix_between(a, b)
    scaled = camera_utils.rotation_matrix_between(a * 1000.0, b * 0.001)
    assert scaled.dtype == dtype
    torch.testing.assert_close(scaled, reference)


@pytest.mark.parametrize("offset", [0.0, 1e-5, 0.5])
def test_parallel_and_nearparallel_rotation_has_finite_correct_jacobian(offset):
    a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.0, offset, 0.0], dtype=torch.float64, requires_grad=True)
    rotation = camera_utils.rotation_matrix_between(a, b)
    assert torch.isfinite(torch.autograd.grad(rotation.sum(), (a, b), retain_graph=True)[0]).all()
    assert torch.autograd.gradcheck(camera_utils.rotation_matrix_between, (a, b), eps=1e-6, atol=2e-5, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direction_mapping_has_normalization_jacobian(dtype):
    a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    b = torch.tensor([0.8, 0.6, 0.0], dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda value: camera_utils.rotation_matrix_between(a, value) @ a, b)
    unit_b = b / b.norm()
    # Since R(a, b) @ unit(a) = unit(b), its Jacobian is independently
    # determined by normalization, including perturbations out of the plane.
    expected = (torch.eye(3, dtype=dtype) - unit_b[:, None] * unit_b[None, :]) / b.norm()
    torch.testing.assert_close(jacobian, expected)
