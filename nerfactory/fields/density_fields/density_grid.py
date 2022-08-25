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
from typing import Callable, List, Tuple, Union

import torch
from torch import nn
from torchtyping import TensorType

import nerfactory.cuda as nerfactory_cuda
from nerfactory.cameras.rays import RaySamples


def _create_grid_coords(resolution: int, device: Union[torch.device, str] = "cpu") -> TensorType["n_coords", 3]:
    """Create 3D grid coordinates

    Args:
        resolution (int): The 3D resolution of the grid.
        device (torch.device): Which device you want the returned tensor to live on.

    Returns:
        TensorType["n_coords", 3]: All grid coordinates with shape [res * res * res, 3].
    """
    arrange = torch.arange(resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid([arrange, arrange, arrange], indexing="ij")
    coords = torch.stack([grid_x, grid_y, grid_z])  # [3, res, res, res]
    coords = coords.reshape(3, -1).t().contiguous()  # [N, 3]
    return coords


class DensityGrid(nn.Module):
    """Cascaded multi-res density grids.

    The DensityGrid contains multi-level (`num_cascades`) grids from finest (0) to the coarsest (num_cascades-1).
    All levels of grid have the same `resolution` but cover different regions in the space. The finest grid is the
    smallest one, which has scale of `base_scale` and centered at `center`. Every next level of grid grows by 2x
    on the scale but all shares the same center.

    For example:
    - `center=0`, `base_scale=3`, `num_cascades=1` would create one grid (lvl=0) covers
            the aabb [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]].
    - `center=0`, `base_scale=3`, `num_cascades=2` would create one grid (lvl=0) covers
            the aabb [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
    and one more grid (lvl=1) covers the aabb [[-3, -3, -3], [3, 3, 3]].

    Normally you can simply set the `center` to the scene center, and `base_scale` to be the scene scale, and use
    `num_cascades=1` to create a single level density grid that covers the scene.

    If, for example, you cares more about the central region of the scene and want to have more fine-grained density
    grid at central region, you can set the `base_scale` to be the scale of the central region that you care. And adjust
    `num_cascades` so that the coarsest level grid covers your entire scene. You should follow this rule so that the
    multi-level grids can cover your scene:

    scene_scale <= base_scale * 2 ** (num_cascades - 1)

    TODO(ruilongli): support `center` and `base_scale` with 3-dim.

    Args:
        center (float): Center of all the grids in the world space. Defaults to 0.
        base_scale (float): Center of the scale of the base level grid. Defaults to 1.
        num_cascades (int): Number of the cascaded multi-res levels. Defaults to 1.
        resolution (int): Resolution of the grid. Defaults to 128.
        update_every_num_iters (int): How frequently to update the grid values. Defaults to 16.
    """

    density_grid: TensorType["num_cascades", "resolution_cubed"]
    mean_density: TensorType[1]

    def __init__(
        self,
        center: float = 0.0,
        base_scale: float = 1.0,
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
        grid_coords = _create_grid_coords(resolution)
        # TODO(ruilongli): hacky way for now. support cpu version of morton3D?
        grid_indices = nerfactory_cuda.morton3D(grid_coords.to("cuda:0")).to("cpu")
        self.register_buffer("grid_coords", grid_coords)
        self.register_buffer("grid_indices", grid_indices)

        mean_density = torch.zeros(())
        self.register_buffer("mean_density", mean_density)

    @torch.no_grad()
    def get_all_cells(self) -> List[Tuple[TensorType["n_cells"], TensorType["n_cells", 3]]]:
        """Returns all cells of the grid.

        Returns:
            List[Tuple[TensorType["n_cells"], TensorType["n_cells", 3]]]: All grid indices and grid
            coordinates of all cascade levels.
        """
        return [(self.grid_indices, self.grid_coords)] * self.num_cascades

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, n: int) -> List[Tuple[TensorType["n"], TensorType["n", 3]]]:
        """Samples both n uniform and occupied cells (per cascade)

        Args:
            n: Number of cells to be sampled.

        Returns:
            List[Tuple[TensorType["n"], TensorType["n", 3]]]: Sampled grid indices and grid
            coordinates of all cascade levels.
        """
        device = self.density_grid.device

        cells = []
        for c in range(self.num_cascades):
            uniform_coords = torch.randint(self.resolution, (n, 3), device=device)
            uniform_indices = nerfactory_cuda.morton3D(uniform_coords)

            occupied_indices = torch.nonzero(self.density_grid[c] > 0)[:, 0]
            if n < len(occupied_indices):
                selector = torch.randint(len(occupied_indices), (n,), device=device)
                occupied_indices = occupied_indices[selector]
            occupied_coords = nerfactory_cuda.morton3D_invert(occupied_indices)

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
    ) -> None:
        """Update the density grid in EMA way.

        Args:
            density_eval_func: A Callable function that takes in sample positions (N, 3) and
                returns densities (N, 1).
            step: The current training step. Instant-NGP has a warmup stage where all cells are updated.
                After certain amount of steps (256), it speeds up by only update sampled cells.
            density_threshold (float): The threshold to prune cells. Recommended to calculate it using this rule:
                `weight_threshold / dt`. For example for Instant-NGP on Lego, the minimum step size `dt` is
                `3` (scale of the scene) * `sqrt(3)` / `1024` (number of samples). And if we want to the weight
                threshold to be `0.01`, we will get `0.01 / (3 * sqrt(3) / 1024) ~= 2.` for density threshold.
            decay (float): EMA decay for density updating.
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
        self.mean_density.data = self.density_grid.mean()

        # pack to bitfield
        self.density_bitfield.data = nerfactory_cuda.packbits(
            self.density_grid, min(self.mean_density.item(), density_threshold)  # type: ignore
        )

        # TODO(ruilongli): max pooling? https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu#L578

    @torch.no_grad()
    def get_densities(self, ray_samples: RaySamples) -> TensorType[..., 1]:
        """Get the density values stored in the density grid.
        Args:
            ray_samples RaySamples: ray samples
        Returns:
            TensorType[..., 1]: Density values
        """
        positions = ray_samples.frustums.get_positions()
        deltas = ray_samples.frustums.ends - ray_samples.frustums.starts

        _mip_levels, indices, _occupancies = nerfactory_cuda.occupancy_query(
            positions.reshape(-1, 3),
            deltas.reshape(-1),
            self.center,
            self.base_scale,
            self.num_cascades,
            self.resolution,
            self.density_bitfield,
        )

        densities = self.density_grid.reshape(-1)[indices.long()]
        densities = densities.view(*positions.shape[:-1], 1)
        return densities

    def forward(self, x):  # pylint: disable=no-self-use
        """Not implemented."""
        raise RuntimeError("Shouldn't be called!")
