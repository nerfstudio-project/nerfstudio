"""
Test samplers
"""
import torch
from pyrad.nerf.sampler import PDFSampler, UniformSampler

from pyrad.structures.rays import RayBundle
from pyrad.nerf.colliders import NearFarCollider


def test_uniform_sampler():
    """Test uniform sampler"""
    near_plane = 2
    far_plane = 4
    num_samples = 15
    sampler = UniformSampler(num_samples=num_samples)

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    ray_bundle = RayBundle(origins=origins, directions=directions)
    collider = NearFarCollider(near_plane, far_plane)
    ray_bundle = collider(ray_bundle)

    ray_samples = sampler(ray_bundle)

    assert ray_samples.positions.shape[-2] == num_samples

    # TODO Tancik: Add more precise tests


def test_pdf_sampler():
    """Test pdf sampler"""
    near_plane = 2
    far_plane = 4
    num_samples = 15

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    ray_bundle = RayBundle(origins=origins, directions=directions)
    collider = NearFarCollider(near_plane, far_plane)
    ray_bundle = collider(ray_bundle)

    uniform_sampler = UniformSampler(num_samples=num_samples)
    coarse_ray_samples = uniform_sampler(ray_bundle)

    weights = torch.ones((10, num_samples))

    # Just check that it doesn't crash. Need to add some actual tests.
    pdf_sampler = PDFSampler(num_samples)
    pdf_sampler(ray_bundle, coarse_ray_samples, weights, num_samples)

    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_uniform_sampler()
    test_pdf_sampler()
