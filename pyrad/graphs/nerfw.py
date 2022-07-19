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
NeRF-W (NeRF in the wild) implementation.
"""

import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure

from pyrad.fields.modules.encoding import NeRFEncoding
from pyrad.fields.modules.field_heads import FieldHeadNames
from pyrad.fields.nerf_field import NeRFField
from pyrad.fields.nerfw_field import VanillaNerfWField
from pyrad.graphs.base import Graph
from pyrad.optimizers.loss import MSELoss
from pyrad.graphs.modules.ray_sampler import PDFSampler, UniformSampler
from pyrad.renderers.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, UncertaintyRenderer
from pyrad.utils import colors
from pyrad.cameras.rays import RayBundle
from pyrad.utils import visualization, writer


class NerfWGraph(Graph):
    """NeRF-W graph"""

    def __init__(
        self,
        intrinsics=None,
        camera_to_world=None,
        near_plane=2.0,
        far_plane=6.0,
        num_coarse_samples=64,
        num_importance_samples=64,
        uncertainty_min=0.03,
        **kwargs,
    ) -> None:
        """A NeRF-W graph.

        Args:
            ...
            uncertainty_min (float, optional): This is added to the end of the uncertainty
                rendering operation. It's called 'beta_min' in other repos.
                This avoids calling torch.log() on a zero value, which would be undefined.
                Defaults to 0.03.
        """
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.num_coarse_samples = num_coarse_samples
        self.num_importance_samples = num_importance_samples
        self.uncertainty_min = uncertainty_min
        self.field_coarse = None
        self.field_fine = None
        self.num_images = len(intrinsics)
        self.appearance_embedding_dim = 48
        self.transient_embedding_dim = 16
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
        self.field_fine = VanillaNerfWField(
            num_images=self.num_images,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            appearance_embedding_dim=self.appearance_embedding_dim,
            transient_embedding_dim=self.transient_embedding_dim,
        )

    def populate_misc_modules(self):
        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.BLACK)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self):
        param_groups = {}
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # coarse field
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)

        # fine weights
        weights_fine = ray_samples_pdf.get_weights(
            field_outputs_fine[FieldHeadNames.DENSITY] + field_outputs_fine[FieldHeadNames.TRANSIENT_DENSITY]
        )
        weights_fine_static = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        weights_fine_transient = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.TRANSIENT_DENSITY])

        # rgb
        rgb_fine_static_component = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        rgb_fine_transient_component = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.TRANSIENT_RGB],
            weights=weights_fine,
        )
        rgb_fine = rgb_fine_static_component + rgb_fine_transient_component
        rgb_fine_static = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine_static,
        )

        # density
        density_transient = field_outputs_fine[FieldHeadNames.TRANSIENT_DENSITY]

        # depth
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        depth_fine_static = self.renderer_depth(weights_fine_static, ray_samples_pdf)

        # uncertainty
        uncertainty = self.renderer_uncertainty(field_outputs_fine[FieldHeadNames.UNCERTAINTY], weights_fine_transient)
        uncertainty += self.uncertainty_min

        outputs = {
            "rgb_coarse": rgb_coarse,  # (num_rays, 3)
            "rgb_fine": rgb_fine,
            "rgb_fine_static": rgb_fine_static,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "depth_fine_static": depth_fine_static,
            "density_transient": density_transient,  # (num_rays, num_samples, 1)
            "uncertainty": uncertainty,  # (num_rays, 1)
        }
        return outputs

    def get_loss_dict(self, outputs, batch):
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        density_transient = outputs["density_transient"]
        betas = outputs["uncertainty"]
        rgb_loss_coarse = 0.5 * ((image - rgb_coarse) ** 2).sum(-1).mean()
        rgb_loss_fine = 0.5 * (((image - rgb_fine) ** 2).sum(-1) / (betas[..., 0] ** 2)).mean()
        uncertainty_loss = 0.5 * (3 + torch.log(betas)).mean()
        density_loss = density_transient.mean()

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "uncertainty_loss": uncertainty_loss,
            "density_loss": density_loss,
        }
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        rgb_fine_static = outputs["rgb_fine_static"]
        depth_coarse = outputs["depth_coarse"]
        depth_fine = outputs["depth_fine"]
        depth_fine_static = outputs["depth_fine_static"]
        uncertainty = outputs["uncertainty"]

        depth_coarse = visualization.apply_depth_colormap(depth_coarse)
        depth_fine = visualization.apply_depth_colormap(depth_fine)
        depth_fine_static = visualization.apply_depth_colormap(depth_fine_static)
        uncertainty = visualization.apply_depth_colormap(uncertainty)

        row0 = torch.cat([image, uncertainty, torch.ones_like(rgb_fine)], dim=-2)
        row1 = torch.cat([rgb_fine, rgb_fine_static, rgb_coarse], dim=-2)
        row2 = torch.cat([depth_fine, depth_fine_static, depth_coarse], dim=-2)
        combined_image = torch.cat([row0, row1, row2], dim=-3)

        writer.put_image(name=f"img/image_idx_{image_idx}-nerfw", image=combined_image, step=step)

        if "mask" in batch:
            mask = batch["mask"].repeat(1, 1, 3)
            writer.put_image(name=f"mask/image_idx_{image_idx}", image=mask, step=step)
