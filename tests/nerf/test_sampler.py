"""
Test samplers
"""
import torch
from mattport.nerf.sampler import PDFSampler, UniformSampler

from mattport.structures.rays import RayBundle


def test_uniform_sampler():
    """Test uniform sampler"""
    near_plane = 2
    far_plane = 4
    num_samples = 15
    sampler = UniformSampler(near_plane=near_plane, far_plane=far_plane, num_samples=num_samples)

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    ray_bundle = RayBundle(origins=origins, directions=directions)

    ray_samples = sampler(ray_bundle)

    assert ray_samples.bins.shape[-1] == num_samples + 1

    # TODO Tancik: Add more precise tests


def test_pdf_sampler():
    """Test pdf sampler"""
    near_plane = 2
    far_plane = 4
    num_samples = 15

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    ray_bundle = RayBundle(origins=origins, directions=directions)

    density = torch.ones((10, num_samples, 1))

    uniform_sampler = UniformSampler(near_plane=near_plane, far_plane=far_plane, num_samples=num_samples)
    coarse_ray_samples = uniform_sampler(ray_bundle)

    # Just check that it doesn't crash. Need to add some actual tests.
    pdf_sampler = PDFSampler(num_samples)
    pdf_sampler(ray_bundle, coarse_ray_samples, density, num_samples)

    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_uniform_sampler()
    test_pdf_sampler()
