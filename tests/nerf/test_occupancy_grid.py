"""
Code to test the occupancy grid.
"""

import torch
from radiance.nerf.occupancy_grid import OccupancyGrid


def test_occupancy_grid():
    """For testing the occupancy grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cascades = 1
    resolution = 128
    aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32).to(device)
    grid = OccupancyGrid(num_cascades, resolution, aabb)
    grid.to(device)
    density_fn = lambda x: x[..., 0:1]  # xyz (..., 3) => density (...,)
    grid.update_occupancy_grid(density_fn=density_fn)


if __name__ == "__main__":
    test_occupancy_grid()
