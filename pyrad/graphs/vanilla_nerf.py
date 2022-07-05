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
Implementation of vanilla nerf.
"""


from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure

from pyrad.fields.modules.encoding import NeRFEncoding
from pyrad.fields.modules.field_heads import FieldHeadNames
from pyrad.fields.nerf_field import NeRFField
from pyrad.graphs.base import Graph
from pyrad.optimizers.loss import MSELoss
from pyrad.graphs.modules.ray_sampler import PDFSampler, UniformSampler
from pyrad.renderers.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from pyrad.utils import colors
from pyrad.cameras.rays import RayBundle
from pyrad.utils import visualization, writer


class NeRFGraph(Graph):
    """Vanilla NeRF graph

    Args:
        intrinsics (torch.Tensor): Camera intrinsics.
        camera_to_world (torch.Tensor): Camera to world transformation.
        near_plane (float, optional): Where to start sampling points. Defaults to a distance of 2,
        far_plane (float, optional): Where to stop sampling points. Defaults to a distance of 6,
        num_coarse_samples (int, optional): Number of samples in coarse field evaluation. Defaults to 64,
        num_importance_samples(int, optional): Number of samples in fine field evaluation. Defaults to 64,
    """

    def __init__(
        self,
        intrinsics: torch.Tensor = None,
        camera_to_world: torch.Tensor = None,
        near_plane: float = 2.0,
        far_plane: float = 6.0,
        num_coarse_samples: int = 64,
        num_importance_samples: int = 128,
        **kwargs,
    ) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_coarse_samples = num_coarse_samples
        self.num_importance_samples = num_importance_samples
        self.field_coarse = None
        self.field_fine = None
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """Set the fields."""

        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

    def populate_misc_modules(self):
        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        image = batch["image"]
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine}
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = visualization.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = visualization.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = visualization.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )
        depth_fine = visualization.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.near_plane,
            far_plane=self.far_plane,
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        writer.put_image(name=f"img/image_idx_{image_idx}", image=combined_rgb, step=step)
        writer.put_image(name=f"accumulation/image_idx_{image_idx}", image=combined_acc, step=step)
        writer.put_image(name=f"depth/image_idx_{image_idx}", image=combined_depth, step=step)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        writer.put_scalar(name=f"psnr/val_{image_idx}-coarse", scalar=float(coarse_psnr), step=step)
        writer.put_scalar(name=f"psnr/val_{image_idx}-fine", scalar=float(fine_psnr), step=step)
        writer.put_scalar(name=f"ssim/val_{image_idx}", scalar=float(fine_ssim), step=step)
        writer.put_scalar(name=f"lpips/val_{image_idx}", scalar=float(fine_lpips), step=step)

        writer.put_scalar(name=writer.EventName.CURR_TEST_PSNR, scalar=float(fine_psnr), step=step)

        return fine_psnr.item()
