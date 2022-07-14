import torch
import torch.nn.functional as F

import pyrad.cuda as pyrad_cuda

# Issue of the F.grid_sample:
# `padding_mode==zeros` means padding zero voxel values outside the grid.
# a query point that is slightly outside the grid would be interpolated
# by the voxel values on the boarder of the grid and the outside zero voxels.
# So it gives non-zero value at regions slightly outside the grid.

grid = torch.ones((1, 1, 128, 128, 128))
positions = torch.tensor([[0.5, 1.004, 0.5]])
values = F.grid_sample(
    grid,
    positions.view(1, -1, 1, 1, 3),
    align_corners=True,
    padding_mode="zeros",
)
print(values.flatten())  # >> 0.7460

# Check out cuda impl is the same with pytorch
device = "cuda:0"
num_cascades = 10
resolution = 128
num_samples = 100_000

# our cuda impl.
density_grid = pyrad_cuda.DensityGrid()
density_grid.num_cascades = num_cascades
density_grid.resolution = resolution
density_grid.aabb = torch.randn((6,), device=device)
density_grid.data = torch.randn((resolution, resolution, resolution, num_cascades), device=device)
positions = torch.randn((num_samples, 3), device=device)
outputs = pyrad_cuda.grid_sample(positions, density_grid)

# pytorch impl.
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
selector = ((positions >= 0) & (positions <= 1)).all(dim=-1)
values *= selector[:, None]

print("is all close:", torch.isclose(values, outputs, atol=1e-5).all(), "max abs err:", (values - outputs).abs().max())
