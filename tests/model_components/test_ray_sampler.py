"""
Test samplers
"""

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.ray_samplers import (
    LinearDisparitySampler,
    LogSampler,
    PDFSampler,
    SqrtSampler,
    UniformSampler,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider


def test_uniform_sampler():
    """Test uniform sampler"""
    num_samples = 15
    sampler = UniformSampler(num_samples=num_samples)

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    radius = torch.ones((10, 1))
    ray_bundle = RayBundle(origins=origins, directions=directions, pixel_area=radius)
    collider = NearFarCollider(near_plane=2, far_plane=4)
    ray_bundle = collider(ray_bundle)

    ray_samples = sampler(ray_bundle)

    assert ray_samples.frustums.get_positions().shape[-2] == num_samples

    # TODO Tancik: Add more precise tests


def test_lin_disp_sampler():
    """Test linear in disparity sampler"""
    num_samples = 15
    sampler = LinearDisparitySampler(num_samples=num_samples)

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    radius = torch.ones((10, 1))
    ray_bundle = RayBundle(origins=origins, directions=directions, pixel_area=radius)
    collider = NearFarCollider(near_plane=2, far_plane=4)
    ray_bundle = collider(ray_bundle)

    ray_samples = sampler(ray_bundle)

    assert ray_samples.frustums.get_positions().shape[-2] == num_samples


def test_sqrt_sampler():
    """Test square root sampler"""
    num_samples = 15
    sampler = SqrtSampler(num_samples=num_samples)

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    radius = torch.ones((10, 1))
    ray_bundle = RayBundle(origins=origins, directions=directions, pixel_area=radius)
    collider = NearFarCollider(near_plane=2, far_plane=4)
    ray_bundle = collider(ray_bundle)

    ray_samples = sampler(ray_bundle)

    assert ray_samples.frustums.get_positions().shape[-2] == num_samples


def test_log_sampler():
    """Test log sampler"""
    num_samples = 15
    sampler = LogSampler(num_samples=num_samples)

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    radius = torch.ones((10, 1))
    ray_bundle = RayBundle(origins=origins, directions=directions, pixel_area=radius)
    collider = NearFarCollider(near_plane=2, far_plane=4)
    ray_bundle = collider(ray_bundle)

    ray_samples = sampler(ray_bundle)

    assert ray_samples.frustums.get_positions().shape[-2] == num_samples


def test_pdf_sampler():
    """Test pdf sampler"""
    num_samples = 15

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    radius = torch.ones((10, 1))
    ray_bundle = RayBundle(origins=origins, directions=directions, pixel_area=radius)
    collider = NearFarCollider(near_plane=2, far_plane=4)
    ray_bundle = collider(ray_bundle)

    uniform_sampler = UniformSampler(num_samples=num_samples)
    coarse_ray_samples = uniform_sampler(ray_bundle)

    weights = torch.ones((10, num_samples, 1))

    # Just check that it doesn't crash. Need to add some actual tests.
    pdf_sampler = PDFSampler(num_samples)
    pdf_sampler(ray_bundle, coarse_ray_samples, weights, num_samples)

    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_uniform_sampler()
    test_pdf_sampler()
