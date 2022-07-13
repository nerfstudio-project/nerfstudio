import torch
import torch.nn.functional as F
import tqdm
from torch.profiler import ProfilerActivity, profile, record_function

import pyrad.cuda as pyrad_cuda

num_rays = 100
num_samples = 64
device = "cuda:8"

# ray_bundle = pyrad_cuda.RayBundle()
# ray_bundle.origins = torch.randn((100, 3), device=device)
# ray_bundle.directions = torch.randn((100, 3), device=device)
# ray_bundle.pixel_area = torch.randn((100, 1), device=device)

# density_grid = pyrad_cuda.DensityGrid()
# density_grid.num_cascades = 1
# density_grid.resolution = 32
# density_grid.aabb = torch.randn((6,), device=device)
# density_grid.data = torch.randn((1, 32, 32, 32), device=device)

# output = pyrad_cuda.generate_ray_samples_uniform(ray_bundle, num_samples, density_grid)
# print(output.packed_indices.shape)

num_cascades = 2
resolution = 3

density_grid = pyrad_cuda.DensityGrid()
density_grid.num_cascades = num_cascades
density_grid.resolution = resolution
density_grid.aabb = torch.randn((6,), device=device)
density_grid.data = torch.randn((resolution, resolution, resolution, num_cascades), device=device)

positions = torch.rand((1_000_000, 3), device=device)
outputs = pyrad_cuda.grid_sample(positions, density_grid)

occupancy_grid = density_grid.data[None, ...].permute(0, 4, 3, 2, 1)
values = (
    F.grid_sample(
        occupancy_grid,
        positions.view(1, -1, 1, 1, 3) * 2.0 - 1.0,
        align_corners=True,
        padding_mode="zeros",
    )
    .view(num_cascades, -1)
    .t()
)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    torch.cuda.synchronize()
    for _ in range(100):
        with record_function("my impl."):
            outputs = pyrad_cuda.grid_sample(positions, density_grid)
            torch.cuda.synchronize()

    for _ in range(100):
        with record_function("pytorch impl."):
            values = (
                F.grid_sample(
                    occupancy_grid,
                    positions.view(1, -1, 1, 1, 3) * 2.0 - 1.0,
                    align_corners=True,
                    padding_mode="zeros",
                )
                .view(num_cascades, -1)
                .t()
            )
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# # print(values[:10])
# # print(outputs[:10])
# print(torch.isclose(values, outputs, atol=1e-6).all(), (values - outputs).abs().max())
