# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import math
# import random
# import sys
# from datetime import datetime
# from typing import NamedTuple

# import numpy as np
# import torch
# from diff_gaussian_rasterization import (GaussianRasterizationSettings,
#                                          GaussianRasterizer)
# from torch import nn

# # TODO (jake-austin): remove this all outright as we move to our own homebrew gaussian splatting based rasterizer

# def inverse_sigmoid(x):
#     return torch.log(x/(1-x))

# def PILtoTorch(pil_image, resolution):
#     resized_image_PIL = pil_image.resize(resolution)
#     resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
#     if len(resized_image.shape) == 3:
#         return resized_image.permute(2, 0, 1)
#     else:
#         return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

# def strip_lowerdiag(L):
#     uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

#     uncertainty[:, 0] = L[:, 0, 0]
#     uncertainty[:, 1] = L[:, 0, 1]
#     uncertainty[:, 2] = L[:, 0, 2]
#     uncertainty[:, 3] = L[:, 1, 1]
#     uncertainty[:, 4] = L[:, 1, 2]
#     uncertainty[:, 5] = L[:, 2, 2]
#     return uncertainty

# def strip_symmetric(sym):
#     return strip_lowerdiag(sym)

# def build_rotation(r):
#     norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

#     q = r / norm[:, None]

#     R = torch.zeros((q.size(0), 3, 3), device='cuda')

#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]

#     R[:, 0, 0] = 1 - 2 * (y*y + z*z)
#     R[:, 0, 1] = 2 * (x*y - r*z)
#     R[:, 0, 2] = 2 * (x*z + r*y)
#     R[:, 1, 0] = 2 * (x*y + r*z)
#     R[:, 1, 1] = 1 - 2 * (x*x + z*z)
#     R[:, 1, 2] = 2 * (y*z - r*x)
#     R[:, 2, 0] = 2 * (x*z - r*y)
#     R[:, 2, 1] = 2 * (y*z + r*x)
#     R[:, 2, 2] = 1 - 2 * (x*x + y*y)
#     return R

# def build_scaling_rotation(s, r):
#     L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
#     R = build_rotation(r)

#     L[:,0,0] = s[:,0]
#     L[:,1,1] = s[:,1]
#     L[:,2,2] = s[:,2]

#     L = R @ L
#     return L

# def safe_state(silent):
#     old_f = sys.stdout
#     class F:
#         def __init__(self, silent):
#             self.silent = silent

#         def write(self, x):
#             if not self.silent:
#                 if x.endswith("\n"):
#                     old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
#                 else:
#                     old_f.write(x)

#         def flush(self):
#             old_f.flush()

#     sys.stdout = F(silent)

#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.set_device(torch.device("cuda:0"))

# def render_from_dict(data : dict, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
#     for k in ['active_sh_degree', 'max_sh_degree', 'xyz', 'scaling', 'rotation', 'features', 'opacity', 'FoVx', 'FoVy', 'image_height', 'image_width', 'world_view_transform', 'full_proj_transform', 'camera_center']:
#         assert k in data, f'Required key {k} not in {data.keys()}'
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(data['xyz'], dtype=data['xyz'].dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = torch.tan(data['FoVx'] * 0.5).item()
#     tanfovy = torch.tan(data['FoVy'] * 0.5).item()

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(data['image_height']),
#         image_width=int(data['image_width']),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=data['world_view_transform'],
#         projmatrix=data['full_proj_transform'],
#         sh_degree=data['active_sh_degree'],
#         campos=data['camera_center'],
#         prefiltered=False,
#         debug=False
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = data['xyz']
#     means2D = screenspace_points
#     opacity = data['opacity']

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     # if pipe.compute_cov3D_python:
#     #     cov3D_precomp = pc.get_covariance(scaling_modifier)
#     # else:
#     scales = data['scaling']
#     rotations = data['rotation']

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if colors_precomp is None:
#         shs = data['features']
#     else:

#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3D_precomp = cov3D_precomp)

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "radii": radii}


# def render_accumulation_from_dict(viewpoint_camera, pc : dict, bg_color : torch.Tensor, scaling_modifier = 1.0):
#     """
#     TODO (jake-austin): double check that this is actually equivalent to getting the accumulation
#     """
#     override_color = torch.ones_like(pc['xyz'], dtype=pc['xyz'].dtype, requires_grad=True, device="cuda") + 0
#     return render_from_dict(
#         viewpoint_camera,
#         pc,
#         bg_color,
#         scaling_modifier,
#         override_color=override_color
#     )


# def inverse_sigmoid(x):
#     return torch.log(x/(1-x))

# C0 = 0.28209479177387814

# def RGB2SH(rgb):
#     return (rgb - 0.5) / C0

# def SH2RGB(sh):
#     return sh * C0 + 0.5


# def getWorld2View(R, t):
#     Rt = torch.zeros((4, 4))
#     Rt[:3, :3] = R.transpose()
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0
#     return torch.float32(Rt)

# def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
#     Rt = torch.zeros((4, 4)).to(R.device)
#     Rt[:3, :3] = R.transpose(-1,-2)
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0

#     C2W = torch.linalg.inv(Rt)
#     cam_center = C2W[:3, 3]

#     cam_center = (cam_center + translate.to(R.device)) * scale
#     C2W[:3, 3] = cam_center
#     Rt = torch.linalg.inv(C2W)
#     return Rt

# def getProjectionMatrix(znear, zfar, fovX, fovY):
#     tanHalfFovY = torch.tan((fovY / 2))
#     tanHalfFovX = torch.tan((fovX / 2))

#     top = tanHalfFovY * znear
#     bottom = -top
#     right = tanHalfFovX * znear
#     left = -right

#     P = torch.zeros(4, 4).to(fovX.device)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)
#     return P

# def fov2focal(fov, pixels):
#     return pixels / (2 * torch.tan(fov / 2))

# def focal2fov(focal, pixels):
#     return 2*torch.atan(pixels/(2*focal))


# class Camera(nn.Module):
#     def __init__(self, R, T, FoVx, FoVy, height, width, data_device = "cuda"):
#         super(Camera, self).__init__()

#         self.R = R
#         self.T = T
#         self.FoVx = FoVx
#         self.FoVy = FoVy

#         try:
#             self.data_device = torch.device(data_device)
#         except Exception as e:
#             print(e)
#             print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
#             self.data_device = torch.device("cuda")

#         self.image_width = width
#         self.image_height = height

#         self.zfar = 100.0
#         self.znear = 0.01

#         self.trans =torch.tensor([0.0, 0.0, 0.0]).to(self.data_device)
#         self.scale = 1.0

#         self.world_view_transform = getWorld2View2(R, T, self.trans, self.scale).transpose(0, 1).to(self.data_device)
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]

# class MiniCam:
#     def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
#         self.image_width = width
#         self.image_height = height    
#         self.FoVy = fovy
#         self.FoVx = fovx
#         self.znear = znear
#         self.zfar = zfar
#         self.world_view_transform = world_view_transform
#         self.full_proj_transform = full_proj_transform
#         view_inv = torch.inverse(self.world_view_transform)
#         self.camera_center = view_inv[3][:3]




# from math import exp

# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable


# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()

# def l2_loss(network_output, gt):
#     return ((network_output - gt) ** 2).mean()

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.size(-3)
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)

# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

