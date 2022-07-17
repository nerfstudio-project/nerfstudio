import torch

import pyrad.cuda_v2 as pyrad_cuda
from pyrad.cuda_pl.backend import _C as pyrad_cuda_pl

device = "cuda:0"

# data = torch.randn((2**3), device=device)
# threshold = 0.5
# bitfield = pyrad_cuda.packbits(data, threshold)
# print(data)
# print(bitfield)


# rays_o = torch.rand((16, 3), device=device)
# rays_d = torch.rand((16, 3), device=device)
# rays_d = torch.nn.functional.normalize(rays_d)
# aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
# t_mim, t_max = pyrad_cuda.ray_aabb_intersect(rays_o, rays_d, aabb)
# print(t_mim, t_max)

# NOTE(ruilongli): pl's code only works on cuda:0
# centers = ((aabb[0:3] + aabb[3:6]) / 2.0).unsqueeze(0)
# half_sizes = ((aabb[3:6] - aabb[0:3]) / 2.0).unsqueeze(0)
# print(centers, half_sizes)
# hit_cnt, hits_t, _ = pyrad_cuda_pl.ray_aabb_intersect(rays_o, rays_d, centers, half_sizes, 1)
# print(hits_t)

coords = torch.randint(0, 32, (2, 3), device=device)
indices = pyrad_cuda.morton3D(coords.to(torch.int32))
print(indices, indices.dtype)
indices = pyrad_cuda.morton3D(coords.to(torch.long))
print(indices, indices.dtype)

hits_t = pyrad_cuda_pl.morton3D(coords.to(torch.int32))
print(indices, indices.dtype)
hits_t = pyrad_cuda_pl.morton3D(coords.to(torch.long))
print(indices, indices.dtype)
