# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Implementation of vanilla nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.model_components.losses import MSELoss, total_variation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import misc


@dataclass
class MSIModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: MSIModel)
    """target class to instantiate"""
    h: int = 960
    w: int = 1920
    nlayers: int = 1
    nsublayers: int = 1
    dmin: float = 2.0
    dmax: float = 20.0
    pose_src: torch.Tensor = torch.eye(4)
    sigmoid_offset: float = -1.0  # 5.0

    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0, "tv_loss": 0.0})


class MSI_field(nn.Module):
    def __init__(self, nlayers, nsublayers, dmin, dmax, pose, H, W, sigmoid_offset):
        super().__init__()
        self.nlayers = nlayers
        self.nsublayers = nsublayers
        self.dmin = dmin
        self.dmax = dmax
        self.pose = pose

        self.H, self.W = H, W

        self.n_total_layers = self.nlayers * self.nsublayers
        self.planes = 1.0 / torch.linspace(1.0 / self.dmin, 1.0 / self.dmax, self.n_total_layers).cuda()

        self.sigmoid_offset = sigmoid_offset

        self.alpha = Parameter(torch.zeros(self.n_total_layers, 1, H, W).uniform_(-1, 1).cuda(), requires_grad=True)
        self.rgb = Parameter(torch.zeros(self.nlayers, 3, H, W).uniform_(-1, 1).cuda(), requires_grad=True)

    def forward(self, ray_bundle: RayBundle):

        center_src = self.pose[:3, 3]

        intersections, mask = MSIModel.intersect_rays_with_spheres(ray_bundle, center_src, self.planes)

        # make them in MSI space
        xyzs = intersections - center_src  # (N, R, 3)
        # normalize by radius
        xyzs_normalized = xyzs / self.planes.reshape(1, -1, 1)  # (N, R, 3)
        # lats = torch.asin(torch.clamp(xyzs_normalized[..., 1], -1.0, 1.0))  # (N, R)
        # lons = torch.atan2(xyzs_normalized[..., 0], -xyzs_normalized[..., 2])  # (N, R)

        # uvs = torch.stack(
        #     [
        #         lons / torch.pi,
        #         2.0 * lats / torch.pi,
        #     ],
        #     dim=2,
        # )  # (N, R, 2)
        uvs = torch.stack(
            [
                xyzs_normalized[..., 2],
                torch.atan2(xyzs_normalized[..., 1], -xyzs_normalized[..., 0]) / (torch.pi),
            ],
            dim=2,
        )  # (N, R, 2)

        # output_vals = torch.zeros((uvs.shape[0], uvs.shape[1], 3))
        # output_vals[(uvs < 0.0).any(dim=-1)] = torch.tensor([1.0, 0.0, 0.0])
        # output_vals = output_vals.reshape(*ray_bundle_shape, 3)
        # outputs["rgb"] = output_vals

        # return outputs

        # print("uvs", uvs[:, :, 0].min(), uvs[:, :, 0].max(), uvs[:, :, 1].min(), uvs[:, :, 1].max())
        uvs = uvs.permute(1, 0, 2).unsqueeze(1)  # (R, 1, N, 2)
        alphas = F.grid_sample(self.alpha, uvs, align_corners=True)  # (R, 1, 1, N)
        alphas_sig = torch.sigmoid(alphas - self.sigmoid_offset)  # (R, 1, 1, N)
        alphas_sig = alphas_sig.permute(0, 2, 3, 1)  # (R, 1, N, 1)

        # # adding mask # (N, R)
        alphas_sig_clone = alphas_sig.clone()
        mask = mask.permute(1, 0).unsqueeze(1).unsqueeze(-1)
        alphas_sig_clone[~mask] = 0

        alphas_sig = alphas_sig_clone
        # alphas_sig[~mask.reshape((alphas_sig.shape[0], 1, -1, 1))] = 0

        # print("alphas_sig", alphas_sig.min(), alphas_sig.max())
        rgbs = F.grid_sample(
            self.rgb, uvs[:: self.nsublayers], align_corners=True, padding_mode="zeros"
        )  # (L // sublayers, 3, 1, N)
        rgbs = torch.sigmoid(rgbs)
        rgbs = rgbs.permute(0, 1, 3, 2)  # (L // sublayers, 3, N, 1)
        rgbs = rgbs.repeat_interleave(self.nsublayers, dim=0)
        # print("rgbs", rgbs[:, :, 0, 0])
        weight = misc.cumprod(1 - alphas_sig, exclusive=True) * alphas_sig
        # print(weight[:, 0, 0, 0])
        output_vals = torch.sum(weight * rgbs, dim=0, keepdim=True)  # [1, 3, N, 1]
        # outputs[mask.squeeze(-1)] = torch.zeros((3,))

        return output_vals


class MSIModel(Model):
    """MSI model

    CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py nerfacto-msi --data /home/akristoffersen/data/bww/ --vis viewer --viewer.websocket-port=7008

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: MSIModelConfig

    def __init__(
        self,
        config: MSIModelConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.pose = kwargs["pose"].cuda()
        self.dmin = kwargs["dmin"]
        self.dmax = kwargs["dmax"]

        # H = self.config.h
        # W = self.config.w

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        self.msi_field = MSI_field(
            self.config.nlayers,
            self.config.nsublayers,
            self.config.dmin,
            self.config.dmax,
            self.config.pose_src.cuda(),
            self.config.h,
            self.config.w,
            self.config.sigmoid_offset,
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.tv_loss = total_variation
        self.rgb_loss = MSELoss()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["planes"] = list(self.msi_field.parameters())
        print([param.shape for param in param_groups["planes"]])

        return param_groups

    @classmethod
    def intersect_rays_with_spheres(
        cls, rays: RayBundle, center=torch.zeros(3), radii=torch.Tensor([1.0])
    ) -> Tuple[TensorType["num_rays", "num_layers", 3], TensorType["num_rays", "num_layers"]]:
        """Intersect provided rays with multiple spheres

        :param rays: RayBundle
        :param center: (1, 3)
        :param radii: (L,)
        """
        R = radii.shape[-1]
        N = rays.shape[0]
        O, D = rays.origins, rays.directions

        O = O.unsqueeze(1).tile(1, R, 1)
        D = D.unsqueeze(1).tile(1, R, 1)

        # print("O", O.min(), O.max())

        # compute quadratic form coefficients
        ray = O - center
        a = (D**2.0).sum(dim=-1).view(N, R)
        b = 2 * (D * ray).sum(dim=-1).view(N, R)
        c = (ray**2.0).sum(dim=-1).view(N, R) - radii[None] ** 2.0

        # solve for ray intersection with sphere
        discriminant = b**2.0 - 4 * a * c  # (N, R)
        t0 = (-b + discriminant.sqrt()) / (2.0 * a)  # (N, R)
        t1 = (-b - discriminant.sqrt()) / (2.0 * a)  # (N, R)

        # ignore rays that miss (both neg) or intersect from outside (both pos)
        mask = t0 * t1 < 0  # (N, R) - if sign differs, then ray intersects from the inside
        intersection = O + D * t0[:, :, None]  # (N, R, 3)
        return intersection, mask  # (N, R, 3)

    def get_outputs(self, ray_bundle: RayBundle):
        # https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html

        outputs = {}
        ray_bundle_shape = ray_bundle.shape

        ray_bundle = ray_bundle.flatten()

        output_vals = self.msi_field(ray_bundle)

        output_vals = output_vals.reshape(*ray_bundle_shape, 3)
        outputs["rgb"] = output_vals

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)
        # print("image", image.min(), image.max())
        # print("image", image[:5])
        # # print('outputs["rgb"]', outputs["rgb"].shape)
        # print("outputs", outputs["rgb"][:5])
        # print("outputs", outputs["rgb"][:5])
        # print("image", image[:5])
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        # tv_loss = self.tv_loss(torch.sigmoid(self.rgb)) + self.tv_loss(torch.sigmoid(self.alpha - self.sigmoid_offset))

        # grad_loss = (torch.mean(torch.abs(ox - gx)) + torch.mean(torch.abs(oy - gy)))

        loss_dict = {"rgb_loss": rgb_loss}

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        print("test 1")
        ssim = self.ssim(image, rgb)
        print("test 2")
        # lpips = self.lpips(image, rgb)
        # print("test 3")

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),  # type: ignore
            # "lpips": float(lpips),
        }
        images_dict = {"img": combined_rgb}
        return metrics_dict, images_dict
