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
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import total_variation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc


@dataclass
class MSIModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: MSIModel)
    """target class to instantiate"""
    h: int = 960
    w: int = 1920
    nlayers: int = 16
    nsublayers: int = 2
    dmin: float = 0.5
    dmax: float = 200.0
    pose_src: torch.Tensor = torch.eye(4)

    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 300
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    """specifies a list of iteration step numbers to perform upsampling"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0, "gradloss": 0.05, "coeff_tv": 0.01})


class MSIModel(Model):
    """Vanilla NeRF model

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
        self.pose = kwargs["pose"]
        self.dmin = kwargs["dmin"]
        self.dmax = kwargs["dmax"]
        self.nlayers = self.config.nlayers
        self.nsublayers = self.config.nsublayers
        self.n_total_layers = self.nlayers * self.nsublayers
        self.planes = 1.0 / torch.linspace(1.0 / self.dmin, 1.0 / self.dmax, self.n_total_layers)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        H = self.config.h
        W = self.config.w

        self.alpha = Parameter(
            torch.zeros(self.n_total_layers, 1, H, W).uniform_(-1, 1),
        )

        self.rgb = Parameter(
            torch.zeros(self.nlayers, 3, H, W).uniform_(-1, 1),
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.tv_loss = total_variation

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["alpha_planes"] = list(self.alpha)
        param_groups["rgb_planes"] = list(self.rgb)

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html

        outputs = {}

        # ray-sphere intersection
        center_src = self.pose[:, :3]
        origin_to_center = center_src - ray_bundle.origins
        t = torch.sum(origin_to_center * ray_bundle.directions, axis=-1)  # type: ignore
        Pe = ray_bundle.origins + ray_bundle.directions * t
        d = torch.linalg.norm(Pe - center_src, axis=-1)

        uvs = torch.zeros((self.n_total_layers, *(ray_bundle.shape[:-1]), 2))
        for i, radius in enumerate(self.planes):
            intersections = torch.sqrt(-1 * d**2 - radius**2)

            xyzs = ray_bundle.origins + ray_bundle.directions * (t + intersections)
            xyzs = (xyzs - center_src) / radius
            lats = torch.asin(torch.clamp(xyzs[..., 1], -1.0, 1.0))
            lons = torch.atan2(xyzs[..., 0], xyzs[..., 2])

            uv = torch.stack(
                [
                    lons / torch.pi,
                    2.0 * lats / torch.pi,
                ]
            )
            uvs[i] = uv

        # then do the sampling

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)

        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
