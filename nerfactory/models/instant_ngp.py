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
Implementation of Instant NGP.
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import nerfactory.cuda as nerfactory_cuda
from nerfactory.cameras.rays import RayBundle
from nerfactory.fields.instant_ngp_field import field_implementation_to_class
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.models.base import Model
from nerfactory.models.modules.ray_sampler import NGPSpacedSampler
from nerfactory.optimizers.loss import MSELoss
from nerfactory.utils import colors, misc, visualization, writer
from nerfactory.utils.callbacks import Callback


class NGPModel(Model):
    """Instant NGP model

    Args:
        field_implementation (str): one of "torch" or "tcnn", or other fields in 'field_implementation_to_class'
        kwargs: additional params to pass up to the parent class model
    """

    def __init__(self, field_implementation="torch", **kwargs) -> None:
        assert field_implementation in field_implementation_to_class
        self.field_implementation = field_implementation
        self.field = None
        super().__init__(**kwargs)

    def get_training_callbacks(self) -> List[Callback]:
        assert self.density_field is not None
        return [
            Callback(
                update_every_num_iters=self.density_field.update_every_num_iters,
                func=self.density_field.update_density_grid,
                density_eval_func=self.field.density_fn,  # type: ignore
            )
        ]

    def populate_fields(self):
        """Set the fields."""
        # torch or tiny-cuda-nn version
        self.field = field_implementation_to_class[self.field_implementation](self.scene_bounds.aabb)

    def populate_misc_modules(self):
        # samplers
        self.sampler = NGPSpacedSampler(num_samples=1024, density_field=self.density_field)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    @torch.cuda.amp.autocast()
    def get_outputs(self, ray_bundle: RayBundle):
        # TODO(ruilongli)
        # - train test difference
        # - visualize "depth_density_grid"
        num_rays = len(ray_bundle)
        device = ray_bundle.origins.device

        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        ray_samples, packed_info, t_min, t_max = self.sampler(ray_bundle, self.field.aabb)

        field_outputs = self.field.forward(ray_samples)
        rgbs = field_outputs[FieldHeadNames.RGB]
        sigmas = field_outputs[FieldHeadNames.DENSITY]

        # accumulate all the rays start from zero opacity
        opacities = torch.zeros((num_rays, 1), device=device)
        (
            accumulated_weight,
            accumulated_depth,
            accumulated_color,
            alive_ray_mask,
        ) = nerfactory_cuda.VolumeRenderer.apply(
            packed_info,
            ray_samples.frustums.starts,
            ray_samples.frustums.ends,
            sigmas.contiguous(),
            rgbs.contiguous(),
            opacities,
        )
        accumulated_depth = torch.clip(accumulated_depth, t_min[:, None], t_max[:, None])
        accumulated_color = accumulated_color + colors.WHITE.to(accumulated_color) * (1.0 - accumulated_weight)

        outputs = {
            "rgb": accumulated_color,
            "accumulation": accumulated_weight,
            "depth": accumulated_depth,
            "alive_ray_mask": alive_ray_mask,  # the rays we kept from sampler
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict, loss_coefficients):
        device = self.device
        image = batch["image"].to(device)
        if "alive_ray_mask" in outputs:
            mask = outputs["alive_ray_mask"]
            rgb_loss = self.rgb_loss(image[mask], outputs["rgb"][mask])
        else:
            rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, loss_coefficients)
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        image = batch["image"]
        rgb = outputs["rgb"]
        acc = visualization.apply_colormap(outputs["accumulation"])
        depth = visualization.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        writer.put_image(name=f"img/image_idx_{image_idx}", image=combined_rgb, step=step)
        writer.put_image(name=f"accumulation/image_idx_{image_idx}", image=combined_acc, step=step)
        writer.put_image(name=f"depth/image_idx_{image_idx}", image=combined_depth, step=step)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        writer.put_scalar(name=f"psnr/val_{image_idx}-fine", scalar=float(psnr), step=step)
        writer.put_scalar(name=f"ssim/val_{image_idx}", scalar=float(ssim), step=step)  # type: ignore
        writer.put_scalar(name=f"lpips/val_{image_idx}", scalar=float(lpips), step=step)

        writer.put_scalar(name=writer.EventName.CURR_TEST_PSNR, scalar=float(psnr), step=step)

        return psnr.item()
