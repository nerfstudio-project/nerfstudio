"""
Test samplers
"""
from random import uniform
import pytest
import torch

from mattport.nerf import sampler
from mattport.structures.cameras import Rays


def test_uniform_sampler():
    """Test uniform sampler"""
    uniform_sampler = sampler.UniformSampler()

    origins = torch.zeros((10, 3))
    directions = torch.ones_like(origins)
    camera_rays = Rays(origin=origins, direction=directions)


def test_pdf_sampler():
    """Test uniform sampler"""
    # TODO Tancik: Add more precise tests


if __name__ == "__main__":
    test_uniform_sampler()
