"""
Test the CUDA bindings.
"""
import torch
from pyrad.cuda.sampler import Sampler
import time
import pyrad_cuda
import os


def test_gradients():
    """_summary_"""
    pass


def test_outputs():
    """_summary_"""
    pass


def pytorch_sample_uniformly_along_ray_bundle(origins, directions, nears, fars, range_offsets):
    """Sample along every ray with a constant step size."""
    time_steps = nears[:, None] + range_offsets[None, :]  # (num_rays, max_num_samples)
    samples = origins[:, None, :] + directions[:, None, :] * time_steps[:, :, None]
    time_steps_mask = (time_steps <= fars[:, None]).float()
    return time_steps, samples, time_steps_mask


def get_rgb_density(samples, time_steps_mask):
    num_rays, max_num_samples = time_steps_mask.shape
    total_max_num_samples = num_rays * max_num_samples

    time_steps_mask_flat = time_steps_mask.view(total_max_num_samples) == 1.0
    samples_flat = samples.view(total_max_num_samples, 3)
    samples_flat_masked = samples_flat[time_steps_mask_flat]

    # TODO: this goes through the forward pass of the MLP

    rgb = torch.zeros((num_rays, max_num_samples, 3))
    density = torch.zeros((num_rays, max_num_samples))
    rgb.view(total_max_num_samples, 3)[time_steps_mask_flat] = 1.0
    density.view(total_max_num_samples)[time_steps_mask_flat] = 1.0
    return rgb, density


def test_benchmark():
    """The test cases assume that rays can be at most of length 1.0."""

    device = torch.device("cuda")
    dtype = torch.float32

    near_range = [0, 0.2]
    far_range = [0.8, 1.0]
    max_num_samples = 128 * 8
    offset = 1.0 / max_num_samples

    print("offset is", offset)
    # import sys; sys.exit()

    print("Running ray sampling tests.")
    # origins = torch.tensor()
    # we want these to be dtype=torch.float32 and device="cuda"
    num_rays = 1024 * 8
    print("num_rays:", num_rays)
    print("max_num_samples:", max_num_samples)
    origins = torch.zeros((num_rays, 3)).to(device)
    directions = torch.ones((num_rays, 3)).to(device)
    nears = torch.rand((num_rays)).to(device) * (near_range[1] - near_range[0]) + near_range[0]
    fars = torch.rand((num_rays)).to(device) * (far_range[1] - far_range[0]) + far_range[0]
    offsets = torch.ones(max_num_samples).to(device) * offset
    range_offsets = torch.arange(0, max_num_samples).to(device) * offset

    start = time.time()
    time_steps_pt, samples_pt, time_steps_mask_pt = pytorch_sample_uniformly_along_ray_bundle(
        origins, directions, nears, fars, range_offsets
    )
    PYTORCH_SAMPLE_TIME = time.time() - start
    print("PYTORCH_SAMPLE_TIME:", PYTORCH_SAMPLE_TIME)

    start = time.time()
    time_steps_cu, samples_cu, time_steps_mask_cu = pyrad_cuda.sample_uniformly_along_ray_bundle(
        origins, directions, nears, fars, offsets, max_num_samples
    )
    CUDA_SAMPLE_TIME = time.time() - start
    print("CUDA_SAMPLE_TIME:   ", CUDA_SAMPLE_TIME)

    # assert torch.allclose(time_steps_pt, time_steps_cu)
    # assert torch.allclose(samples_pt, samples_cu)
    # assert torch.allclose(time_steps_mask_pt, time_steps_mask_cu)

    # rgb_pt, density_pt = get_rgb_density(samples_pt, time_steps_mask_pt)
    # rgb_cu, density_cu = get_rgb_density(samples_cu, time_steps_mask_cu)

    perc_samples_used = float(time_steps_mask_pt.sum() / (time_steps_mask_pt.shape[0] * time_steps_mask_pt.shape[1]))
    perc_samples_used_str = str(round(perc_samples_used * 100.0, 2))
    print(f"Using field network for %{perc_samples_used_str} of samples in this ray bundle.")

    # TODO: rendering of rgb, density, and deltas with pytorch vs. cuda


if __name__ == "__main__":
    test_benchmark()
