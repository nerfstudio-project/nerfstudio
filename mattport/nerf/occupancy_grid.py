"""
Code to implement the density grid.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from mattport.utils.misc import is_not_none


class OccupancyGrid(nn.Module):
    """Module to keep track of the density and occupancy."""

    def __init__(
        self, num_cascades=1, resolution=128, aabb=None, density_fn=None, update_every_num_iters: int = 16
    ) -> None:
        super().__init__()
        assert is_not_none(aabb), "The axis-aligned bounding box aabb is not defined!"
        self.num_cascades = num_cascades  # the number of levels (i.e, cascades)
        self.resolution = resolution
        self.register_buffer("aabb", aabb)  # axis-aligned bounding box
        occupancy_grid = torch.zeros(
            [self.num_cascades] + [self.resolution] * 3
        )  # (num_cascades, x_resolution, y_resolution, z_resolution)
        # not a module parameter, but it still should be saved and part of `state_dict`
        self.register_buffer("occupancy_grid", occupancy_grid)
        self.mean_density = 0.0

        self.density_fn = density_fn
        self.update_every_num_iters = update_every_num_iters
        self.iteration_count = 0

    def reset(self):
        """Zero out the occupancy grid."""
        self.occupancy_grid.zero_()

    @torch.no_grad()
    def mark_untrained_grid(self):
        """Marks the locations in the density grid that are never seen by training views."""
        # TODO: only query points that are covered by the training cameras viewing frustrum
        # We'll need to pass in the camera poses and intrinsics to this function
        raise NotImplementedError()

    @torch.no_grad()
    def update_occupancy_grid(self, density_fn, decay=0.95, split_size=64):
        """Update the occupancy grid with density values.

        Args:
            density_fn: Function from takes points in the shape (..., 3) and
            returns a tensor of densities of shape (..., 1).
        """
        # TODO(ethan): remove this pylint disable
        #  pylint: disable=invalid-name

        device = self.occupancy_grid.device

        assert self.num_cascades == 1, "Still need to implement code for multiple cascades."
        temp_occupancy_grid = torch.zeros_like(self.occupancy_grid)

        X_length, Y_length, Z_length = self.aabb[1] - self.aabb[0]
        X_half_voxel_length = 0.5 * X_length / self.resolution
        Y_half_voxel_length = 0.5 * Y_length / self.resolution
        Z_half_voxel_length = 0.5 * Z_length / self.resolution
        half_voxel_lengths = torch.tensor(
            [X_half_voxel_length, Y_half_voxel_length, Z_half_voxel_length], device=device
        )
        X = torch.linspace(
            self.aabb[0][0] + X_half_voxel_length, self.aabb[1][0] - X_half_voxel_length, self.resolution, device=device
        ).split(split_size)
        Y = torch.linspace(
            self.aabb[0][1] + Y_half_voxel_length, self.aabb[1][1] - Y_half_voxel_length, self.resolution, device=device
        ).split(split_size)
        Z = torch.linspace(
            self.aabb[0][2] + Z_half_voxel_length, self.aabb[1][2] - Z_half_voxel_length, self.resolution, device=device
        ).split(split_size)
        for x_index, xs in enumerate(X):
            for y_index, ys in enumerate(Y):
                for z_index, zs in enumerate(Z):
                    x_len, y_len, z_len = len(xs), len(ys), len(zs)
                    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
                    xyzs = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (resolution, resolution,, resolution, 3)
                    # add noise to the points
                    xyzs_rand = (
                        torch.rand_like(xyzs) * half_voxel_lengths
                    )  # range [-half_voxel_length, half_voxel_length] for each dimension
                    xyzs += xyzs_rand
                    densities = density_fn(xyzs)
                    assert densities.shape[-1] == 1
                    temp_occupancy_grid[
                        0,
                        x_index * split_size : x_index * split_size + x_len,
                        y_index * split_size : y_index * split_size + y_len,
                        z_index * split_size : z_index * split_size + z_len,
                    ] = densities[..., 0]

        # exponential moving average update
        valid_mask = self.occupancy_grid >= 0.0
        self.occupancy_grid[valid_mask] = torch.maximum(
            self.occupancy_grid[valid_mask] * decay, temp_occupancy_grid[valid_mask]
        )
        self.mean_density = torch.mean(self.occupancy_grid[valid_mask]).item()

    @torch.no_grad()
    def get_densities(self, xyzs: TensorType[..., 3]) -> TensorType[..., 1]:
        """Trilinear interpolation to get the density values.

        Args:
            xyzs (TensorType[..., 3]): 3D querry coordinate

        Returns:
            TensorType[..., 1]: Density values
        """
        occupancy_grid = self.occupancy_grid[None, ...]  # shape (1, num_cascades, X_res, Y_res, Z_res)
        xyzs_shape = xyzs.shape
        xyzs_reshaped = xyzs.view(1, -1, 1, 1, 3)
        voxel_lengths = self.aabb[1] - self.aabb[0]
        xyzs_reshaped_normalized = ((xyzs_reshaped - self.aabb[0]) / voxel_lengths) * 2.0 - 1.0
        densities = F.grid_sample(
            occupancy_grid.permute(
                0, 1, 4, 3, 2
            ),  # (xyz to zyx) for grid sample useage because the xyzs have xyz ordering
            xyzs_reshaped_normalized,
            align_corners=True,
            padding_mode="zeros",
        )
        densities = densities.view(*xyzs_shape[:-1], 1)
        return densities

    def forward(self, x):
        """Not implemented."""
        raise NotImplementedError()
