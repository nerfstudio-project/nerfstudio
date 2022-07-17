import numpy as np
import tinycudann as tcnn
import torch
from torch import nn

from pyrad.cuda_pl.backend import _C as vren

from .custom_functions import TruncExp


class NGP(nn.Module):
    def __init__(self, scale=0.5):
        super().__init__()

        # scene bounding box
        self.scale = scale
        self.register_buffer("center", torch.zeros(1, 3))
        self.register_buffer("xyz_min", -torch.ones(1, 3) * scale)
        self.register_buffer("xyz_max", torch.ones(1, 3) * scale)
        self.register_buffer("half_size", (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.register_buffer(
            "density_bitfield", torch.zeros(self.cascades * self.grid_size**3 // 8, dtype=torch.uint8)
        )

        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16

        self.xyz_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=16,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": np.exp(np.log(2048 * scale / N_min) / (L - 1)),
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

        self.sigma_act = TruncExp.apply

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        h = self.xyz_encoder(x)
        sigmas = self.sigma_act(h[:, 0])
        if return_feat:
            return sigmas, h
        return sigmas

    def forward(self, x, d):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d /= torch.norm(d, dim=-1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M):
        """
        Sample both M uniform and occupied cells (per cascade)

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32, device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c] > 0)[:, 0]
            rand_idx = torch.randint(len(indices2), (M,), device=self.density_grid.device)
            indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        w2c_R = poses[:, :3, :3].mT  # (N, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]  # M=128^3 cells
            for i in range(0, len(indices), chunk):
                xyzs = coords[i : i + chunk] / (self.grid_size - 1) * 2 - 1  # in [-1, 1]
                s = min(2 ** (c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N, 3, chunk)
                uvd = K @ xyzs_c
                uv = uvd[:, :2] / uvd[:, 2:]  # (N, 2, chunk)
                valid_mask = (
                    (uvd[:, 2] > 0)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < img_wh[0])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < img_wh[1])
                )  # (N, chunk)
                self.density_grid[c, indices[i : i + chunk]] = torch.where(valid_mask.any(0), 0.0, -1.0)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95):
        # create temporary grid
        tmp_grid = -torch.ones_like(self.density_grid)
        if warmup:  # during the first 256 steps
            cells = self.get_all_cells()
        else:
            N = self.grid_size**3 // 4
            cells = self.sample_uniform_and_occupied_cells(N)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            xyzs = coords / (self.grid_size - 1) * 2 - 1  # in [-1, 1]
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = xyzs * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            tmp_grid[c, indices] = self.density(xyzs_w)

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        mean_density = self.density_grid[self.density_grid > 0].mean().item()

        # pack to bitfield
        vren.packbits(self.density_grid, min(mean_density, density_threshold), self.density_bitfield)

        # TODO: max pooling? https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu#L578
