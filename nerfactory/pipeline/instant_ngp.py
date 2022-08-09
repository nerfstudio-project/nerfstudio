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
Instant-NGP Pipeline
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfactory.cameras.cameras import Camera

import nerfactory.cuda as nerfactory_cuda
from nerfactory.cameras.rays import RayBundle
from nerfactory.data.dataloader import TestStoredDataloader
from nerfactory.data.structs import SceneBounds
from nerfactory.fields.density_fields.density_grid import DensityGrid
from nerfactory.fields.instant_ngp_field import TorchInstantNGPField
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.graphs.modules.ray_sampler import NGPSpacedSampler
from nerfactory.graphs.modules.scene_colliders import SceneBoundsCollider
from nerfactory.model.base import Model
from nerfactory.optimizers.loss import MSELoss
from nerfactory.pipeline.base import Pipeline
from nerfactory.utils import colors
from nerfactory.utils.misc import get_masked_dict


class InstantNGPModel(Model):
    """
    The model class for the Instant NGP model.
    """

    def __init__(
        self,
        collider: SceneBoundsCollider,
        scene_bounds: SceneBounds = None,
    ) -> None:
        super().__init__(collider, scene_bounds)

    def populate_fields(self):
        self.field = TorchInstantNGPField(self.scene_bounds.aabb)

    def populate_misc_modules(self):
        """Initializes any additional modules that are part of the network."""
        self.density_grid = DensityGrid(center=0.0, base_scale=3, num_cascades=1)

        # samplers
        self.sampler = NGPSpacedSampler(num_samples=1024, density_field=self.density_grid)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Dict[str, List[Parameter]]: Mapping of different parameter groups
        """
        return {"field": list(self.field.parameters())}

    def forward(self, ray_bundle: RayBundle, batch):  # pylint:disable=arguments-differ
        """Run the forward starting with ray indices."""

        if self.collider is not None:
            intersected_ray_bundle = self.collider(ray_bundle)
            valid_mask = intersected_ray_bundle.valid_mask[..., 0]
        else:
            # NOTE(ruilongli): we don't need collider for ngp
            intersected_ray_bundle = ray_bundle
            valid_mask = None

        if valid_mask is not None:
            intersected_ray_bundle = intersected_ray_bundle[valid_mask]
            # during training, keep only the rays that intersect the scene. discard the rest
            batch = get_masked_dict(batch, valid_mask)  # NOTE(ethan): this is really slow if on CPU!

        ray_bundle = intersected_ray_bundle

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

        return accumulated_color, accumulated_weight, accumulated_depth, alive_ray_mask

    # pylint:disable=arguments-differ
    def get_loss_dict(self, outputs_rgb, batch, outputs_mask=None) -> Dict[str, torch.tensor]:
        """Computes and returns the losses."""
        device = self.get_device()
        image = batch["image"].to(device)
        if outputs_mask is not None:
            rgb_loss = self.rgb_loss(image[outputs_mask], outputs_rgb[outputs_mask])
        else:
            rgb_loss = self.rgb_loss(image, outputs_rgb)
        loss_dict = {"rgb_loss": rgb_loss}
        return loss_dict


class InstantNGPPipeline(Pipeline):
    """
    The pipeline class for the Instant NGP model."""

    dataloader: TestStoredDataloader

    def get_train_loss_dict(self):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the dataloader and interfacing with the
        Model class."""
        rays, batch = self.dataloader_train_iter.next()
        accumulated_color, _, _, mask = self.model(rays, batch)
        masked_batch = get_masked_dict(batch, mask)
        loss_dict = self.model.get_loss_dict(accumulated_color, masked_batch, mask)
        return loss_dict

    def get_eval_loss_dict(self):
        """This function gets your evaluation loss dict. It needs to get the data
        from the dataloader and feed it to the model,"""
        rays, batch = self.dataloader_eval_iter.next()
        accumulated_color, _, _, mask = self.model(rays, batch)
        masked_batch = get_masked_dict(batch, mask)
        loss_dict = self.model.get_loss_dict(accumulated_color, masked_batch, mask)
        return loss_dict

    def test_image_outputs(self) -> None:
        """Log the test image outputs"""
        camera = Camera(self.dataloader.eval_datasetinputs.camera_to_world[0])
