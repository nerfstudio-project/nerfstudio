import torch

from pyrad.cuda_pl.custom_functions import RayAABBIntersector

N_rays = 1024
N_voxels = 100
max_hits = 100
device = "cuda:1"

rays_o = torch.randn((N_rays, 3), device=device)
rays_d = torch.randn((N_rays, 3), device=device)
centers = torch.randn((N_voxels, 3), device=device)
half_sizes = torch.randn((N_voxels, 3), device=device)

hits_cnt, hits_t, hits_voxel_idx = RayAABBIntersector.apply(rays_o, rays_d, centers, half_sizes, max_hits)
