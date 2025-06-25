"""
Test pose utils
"""

import torch

from nerfstudio.utils import poses


def test_to4x4():
    """Test addition of homogeneous coordinate to 3D pose."""
    pose = torch.rand((10, 3, 4))
    pose_4x4 = poses.to4x4(pose)

    assert pose_4x4.shape == (10, 4, 4)
    assert torch.equal(
        pose_4x4[:, 3, 3],
        torch.ones(
            10,
        ),
    )


def test_multiply():
    """Test 3D pose multiplication."""
    pose = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        ]
    )
    translation_pose = poses.multiply(pose, pose)
    assert translation_pose.shape == pose.shape
    assert torch.equal(translation_pose[..., :, 3], torch.tensor([[2.0, 4.0, 6.0]]))

    pose_a = pose.clone()
    pose_a[:, :3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    pose_b = pose.clone()
    pose_b[:, :3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    translation_rotation_pose = poses.multiply(pose_a, pose_b)
    assert torch.allclose(translation_rotation_pose, (poses.to4x4(pose_a) @ poses.to4x4(pose_b))[:, :3, :4])


def test_inverse():
    """Test 3D pose inversion."""

    pose = torch.rand((10, 3, 4))
    pose[:, :3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    pose_inv = poses.inverse(pose)

    assert pose_inv.shape == pose.shape

    unit_pose = torch.zeros_like(pose)
    unit_pose[:, :3, :3] = torch.eye(3)

    assert torch.allclose(
        poses.multiply(pose, pose_inv),
        unit_pose,
    )


def test_normalize():
    """Test 3D pose normalization"""
    pose = torch.ones((2, 3, 4))
    pose[:, :3, :3] = torch.eye(3)
    pose[0, :3, 3] = torch.tensor([2.0, 0.0, -2.0])
    pose[1, :3, 3] = torch.tensor([1.0, 1.0, 1.0])

    pose_scaled = poses.normalize(pose)

    assert pose_scaled.shape == pose.shape
    assert torch.max(pose_scaled[:, :3, 3]) <= 1.0

    assert torch.equal(pose_scaled[0, :3, 3], torch.tensor([1.0, 0.0, -1.0]))
    assert torch.equal(pose_scaled[1, :3, 3], torch.tensor([0.5, 0.5, 0.5]))
