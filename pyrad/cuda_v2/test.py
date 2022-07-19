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

# coords = torch.randint(0, 32, (2, 3), device=device)
# print("coords", coords)

# indices = pyrad_cuda.morton3D(coords.to(torch.int32))
# coords = pyrad_cuda.morton3D_invert(indices.to(torch.int32))
# print(indices, indices.dtype, coords, coords.dtype)
# indices = pyrad_cuda.morton3D(coords.to(torch.long))
# coords = pyrad_cuda.morton3D_invert(indices.to(torch.long))
# print(indices, indices.dtype, coords, coords.dtype)

# indices = pyrad_cuda_pl.morton3D(coords.to(torch.int32))
# coords = pyrad_cuda_pl.morton3D_invert(indices.to(torch.int32))
# print(indices, indices.dtype, coords, coords.dtype)
# indices = pyrad_cuda_pl.morton3D(coords.to(torch.long))
# coords = pyrad_cuda_pl.morton3D_invert(indices.to(torch.long))
# print(indices, indices.dtype, coords, coords.dtype)


rays_o = torch.rand((1, 3), device=device)
rays_d = torch.rand((1, 3), device=device)
rays_d = torch.nn.functional.normalize(rays_d)
aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
t_min, t_max = pyrad_cuda.ray_aabb_intersect(rays_o, rays_d, aabb)
print(t_min, t_max)

cascades = 4
grid_size = 32
density = torch.rand((cascades * grid_size**3), device=device)
threshold = 0.0
bitfield = pyrad_cuda.packbits(density, threshold)
# print("bitfield", bitfield)

num_steps = 16
max_samples = num_steps * rays_o.shape[0]
cone_angle = 0.0  # 1.0 / 256
packed_info, positions, dirs, deltas, ts = pyrad_cuda.raymarching_train(
    rays_o, rays_d, t_min, t_max, cascades, grid_size, bitfield, max_samples, num_steps, cone_angle
)
print(positions)

sigmas = torch.rand_like(deltas)
rgbs = torch.rand_like(positions)
accumulated_weight, accumulated_depth, accumulated_color = pyrad_cuda.volumetric_rendering(
    packed_info, positions, deltas, ts, sigmas, rgbs
)
print(accumulated_weight)
print(accumulated_depth)
print(accumulated_color)

grad_weight = torch.randn_like(accumulated_weight)
grad_depth = torch.randn_like(accumulated_depth) * 0
grad_color = torch.randn_like(accumulated_color)

grad_sigmas, grad_rgbs = pyrad_cuda.volumetric_rendering_backward(
    accumulated_weight,
    accumulated_depth,
    accumulated_color,
    grad_weight,
    grad_depth,
    grad_color,
    packed_info,
    deltas,
    ts,
    sigmas,
    rgbs,
)
print(grad_sigmas)
print(grad_rgbs)


rays_o = rays_o - 0.5
aabb = aabb - 0.5
scale = 0.5
noise = torch.zeros_like(rays_o[:, 0])
centers = ((aabb[0:3] + aabb[3:6]) / 2.0).unsqueeze(0)
half_sizes = ((aabb[3:6] - aabb[0:3]) / 2.0).unsqueeze(0)
_, hits_t, _ = pyrad_cuda_pl.ray_aabb_intersect(rays_o, rays_d, centers, half_sizes, 1)
print(hits_t[:, 0])

rays_a, xyzs, dirs, deltas, ts, counter = pyrad_cuda_pl.raymarching_train(
    rays_o, rays_d, hits_t[:, 0], bitfield.reshape(cascades, -1), scale, cone_angle, noise, grid_size, num_steps
)
# print(xyzs + 0.5)

opacity, depth, rgb = pyrad_cuda_pl.composite_train_fw(sigmas, rgbs, deltas, ts, rays_a, 1e-4)
print(opacity)
print(depth)
print(rgb)

grad_sigmas, grad_rgbs = pyrad_cuda_pl.composite_train_bw(
    grad_weight.squeeze(-1),
    grad_depth.squeeze(-1),
    grad_color,
    sigmas.squeeze(-1),
    rgbs,
    deltas,
    ts,
    rays_a,
    opacity,
    depth,
    rgb,
    1e-4,
)
print(grad_sigmas)
print(grad_rgbs)
