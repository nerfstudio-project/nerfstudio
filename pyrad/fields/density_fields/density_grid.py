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
from torch import nn
from torchtyping import TensorType

import pyrad.cuda as pyrad_cuda


def create_grid_coords(resolution: int, device: torch.device = "cpu") -> TensorType["n_coords", 3]:
    """Create 3D grid coordinates"""
    arrange = torch.arange(resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid([arrange, arrange, arrange], indexing="ij")
    coords = torch.stack([grid_x, grid_y, grid_z])  # [3, steps[0], steps[1], steps[2]]
    coords = coords.reshape(3, -1).t().contiguous()  # [N, 3]
    return coords


class DensityGrid(nn.Module):
    """Cascaded multi-res density grids.

    The multi-res grids are all centerd at [center, center, center] in the world space.
    The first grid covers regions (center - base_scale / 2, center + base_scale / 2)
    and the followings grows by 2x scale each.
    """

    def __init__(
        self,
        center: float = 0.0,
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
        self.update_every_num_iters = update_every_num_iters

        density_grid = torch.zeros([self.num_cascades] + [self.resolution**3])
        self.register_buffer("density_grid", density_grid)

        density_bitfield = torch.zeros([self.num_cascades * self.resolution**3 // 8], dtype=torch.uint8)
        self.register_buffer("density_bitfield", density_bitfield)

        # Integer grid coords / indices that do not related to cascades
        grid_coords = create_grid_coords(resolution)
        # TODO(ruilongli): hacky way for now. support cpu version of morton3D?
        grid_indices = pyrad_cuda.morton3D(grid_coords.to("cuda:0")).to("cpu")
        self.register_buffer("grid_coords", grid_coords)
        self.register_buffer("grid_indices", grid_indices)

    @torch.no_grad()
    def reset(self):
        """Zeros out the density grid."""
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
        self,
        density_eval_func: Callable,
        step: int,
        density_threshold: float = 2,  # 0.01 / (SQRT3 / 1024 * 3)
        decay: float = 0.95,
    ) -> NoReturn:
        """Update the density grid in EMA way.

        Args:
            density_eval_func: A Callable function that takes in samples (N, 3)
                and returns densities (N, 1).
        """
        # create temporary grid
        tmp_grid = -torch.ones_like(self.density_grid)
        if step < 256:
            cells = self.get_all_cells()
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
        mean_density = self.density_grid.mean().item()

        # pack to bitfield
        self.density_bitfield.data = pyrad_cuda.packbits(self.density_grid, min(mean_density, density_threshold))

        # TODO(ruilongli): max pooling? https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu#L578

    @torch.no_grad()
    def get_densities(self, xyzs: TensorType[..., 3]) -> TensorType[..., 1]:
        """Trilinear interpolation to get the density values.
        Args:
            xyzs (TensorType[..., 3]): 3D querry coordinate
        Returns:
            TensorType[..., 1]: Density values
        """
        # TODO(ruilongli): pass in the t along the ray as well as the xyz, so that we
        # can decide mip-level as in NGP.
        raise NotImplementedError("We haven't implement the logic of querying densities from cascaded grid.")
        # density_grid = self.density_grid.reshape(
        #     1, self.num_cascades, self.resolution, self.resolution, self.resolution
        # )  # shape (1, num_cascades, X_res, Y_res, Z_res)
        # xyzs_shape = xyzs.shape
        # xyzs_reshaped = xyzs.view(1, -1, 1, 1, 3)
        # voxel_lengths = self.aabb[1] - self.aabb[0]
        # xyzs_reshaped_normalized = ((xyzs_reshaped - self.aabb[0]) / voxel_lengths) * 2.0 - 1.0
        # densities = F.grid_sample(
        #     density_grid.permute(
        #         0, 1, 4, 3, 2
        #     ),  # (xyz to zyx) for grid sample useage because the xyzs have xyz ordering
        #     xyzs_reshaped_normalized,
        #     align_corners=True,
        #     padding_mode="zeros",
        # )
        # densities = densities.view(*xyzs_shape[:-1], 1)
        # return densities

    def forward(self, x):
        """Not implemented."""
        raise RuntimeError("Shouldn't be called!")
