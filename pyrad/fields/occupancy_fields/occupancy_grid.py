# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to implement the density grid.
"""
from typing import Callable, List, NoReturn, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

import pyrad.cuda_v2 as pyrad_cuda
from pyrad.utils.misc import is_not_none


def create_grid_coords(resolution: int, device: torch.device = "cpu") -> TensorType["n_coords", 3]:
    """Create 3D grid coordinates"""
    arrange = torch.arange(resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid([arrange, arrange, arrange], indexing="ij")
    coords = torch.stack([grid_x, grid_y, grid_z])  # [3, steps[0], steps[1], steps[2]]
    coords = coords.reshape(3, -1).t().contiguous()  # [N, 3]
    return coords


class DensityGrid(nn.Module):
    """Cascaded multi-res density grids.

    The multi-res grids are all centerd at [0.5, 0.5, 0.5] in the world space.
    The first grid covers regions [0., 0., 0.] ~ [1., 1., 1.] and the followings
    grows by 2x scale each.
    """

    def __init__(
        self,
        center: float = 0.5,
        base_scale: float = 1.0,  # base level scale
        num_cascades: int = 1,
        resolution: int = 128,
        update_every_num_iters: int = 16,
    ) -> None:
        super().__init__()
        self.center = center
        self.base_scale = base_scale
        self.num_cascades = num_cascades  # the number of levels (i.e, cascades)
        self.resolution = resolution
        self.mean_density = 0.0
        self.update_every_num_iters = update_every_num_iters

        density_grid = torch.zeros([self.num_cascades] + [self.resolution**3])
        self.register_buffer("density_grid", density_grid)

        density_bitfield = torch.zeros([self.num_cascades * self.resolution**3 // 8], dtype=torch.uint8)
        self.register_buffer("density_bitfield", density_bitfield)

        # Integer grid coords / indices that do not related to cascades
        grid_coords = create_grid_coords(resolution)
        # TODO(ruilongli): hacky way for now. support cpu version?
        grid_indices = pyrad_cuda.morton3D(grid_coords.to("cuda:0")).to("cpu")
        self.register_buffer("grid_coords", grid_coords)
        self.register_buffer("grid_indices", grid_indices)

        self.warmup_steps = 0

    @torch.no_grad()
    def reset(self):
        """Zeros out the occupancy grid."""
        self.density_grid.zero_()
        self.density_bitfield.zero_()

    @torch.no_grad()
    def get_all_cells(self) -> List[Tuple[TensorType["n"], TensorType["n"]]]:
        """Returns all cells"""
        return [(self.grid_indices, self.grid_coords)] * self.num_cascades

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, n: int) -> List[Tuple[TensorType["n"], TensorType["n"]]]:
        """Samples both n uniform and occupied cells (per cascade)"""
        device = self.density_grid.device

        cells = []
        for c in range(self.num_cascades):
            uniform_coords = torch.randint(self.resolution, (n, 3), device=device)
            uniform_indices = pyrad_cuda.morton3D(uniform_coords)

            occupied_indices = torch.nonzero(self.density_grid[c] > 0)[:, 0]
            if n < len(occupied_indices):
                selector = torch.randint(len(occupied_indices), (n,), device=device)
                occupied_indices = occupied_indices[selector]
            occupied_coords = pyrad_cuda.morton3D_invert(occupied_indices)

            # concatenate
            coords = torch.cat([uniform_coords, occupied_coords], dim=0)
            indices = torch.cat([uniform_indices, occupied_indices], dim=0)
            cells.append((indices, coords))
        return cells

    @torch.no_grad()
    def update_density_grid(
        self, density_eval_func: Callable, density_threshold: float = 1e-4, decay: float = 0.95
    ) -> NoReturn:
        """Update the density grid in EMA way.

        Args:
            density_eval_func: A Callable function that takes in samples (N, 3)
                and returns densities (N, 1).
        """
        # create temporary grid
        tmp_grid = -torch.ones_like(self.density_grid)
        if self.warmup_steps < 256:
            cells = self.get_all_cells()
            self.warmup_steps += 1
        else:
            N = self.resolution**3 // 4
            cells = self.sample_uniform_and_occupied_cells(N)

        # infer sigmas
        for mip_level in range(self.num_cascades):
            mip_scale = 2 ** (-mip_level) / self.base_scale
            indices, coords = cells[mip_level]
            # `coords` denotes the i-th cell. It's a poor naming here.
            # the actually coordinates in world space x has mapping to
            # the cell like this:
            # x \in [0, 1/res] maps to 0-th cell.
            # x \in [1 - 1/res, 1] maps to (res-1)-th cell.
            x = (coords + torch.rand_like(coords.float())) / self.resolution
            x = (x - 0.5) / mip_scale + self.center
            tmp_grid[mip_level, indices] = density_eval_func(x).squeeze(-1)

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        mean_density = self.density_grid[self.density_grid > 0].mean().item()

        # pack to bitfield
        self.density_bitfield.data = pyrad_cuda.packbits(self.density_grid, min(mean_density, density_threshold))

        # TODO: max pooling? https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu#L578

    def forward(self, x):
        """Not implemented."""
        raise RuntimeError("Shouldn't be called!")


class OccupancyGrid(nn.Module):
    """Module to keep track of the density and occupancy."""

    def __init__(
        self, num_cascades: int = 1, resolution: int = 128, aabb=None, update_every_num_iters: int = 16
    ) -> None:
        super().__init__()
        assert is_not_none(aabb), "The axis-aligned bounding box aabb is not defined!"
        self.num_cascades = num_cascades  # the number of levels (i.e, cascades)
        self.resolution = resolution
        self.register_buffer("aabb", aabb)  # axis-aligned bounding box
        occupancy_grid = torch.ones(
            [self.num_cascades] + [self.resolution] * 3
        )  # (num_cascades, x_resolution, y_resolution, z_resolution)
        # not a module parameter, but it still should be saved and part of `state_dict`
        self.register_buffer("occupancy_grid", occupancy_grid)
        self.mean_density = 0.0

        self.update_every_num_iters = update_every_num_iters

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

        # TODO(ruilongli): pyrad.cuda.grid_sampler is 1.5x faster, but we might want a
        # switch to toggle between the two. Also need to figure out how to initialize
        # pyrad.cuda.DensityGrid() in __init__()

        # grid_cu = pyrad.cuda.DensityGrid()
        # grid_cu.num_cascades = self.num_cascades
        # grid_cu.resolution = self.resolution
        # grid_cu.aabb = self.aabb
        # grid_cu.data = self.occupancy_grid.permute((1, 2, 3, 0)).contiguous()
        # positions = (xyzs.reshape(-1, 3) - self.aabb[0]) / voxel_lengths
        # densities = pyrad.cuda.grid_sample(positions, self.grid_cu)
        # densities = densities.view(*xyzs_shape[:-1], 1)

        return densities

    def forward(self, x):
        """Not implemented."""
        raise NotImplementedError()
