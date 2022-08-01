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
TensorRF implementation.
"""

from typing import Dict, List

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfactory.cameras.rays import RayBundle
from nerfactory.fields.modules.encoding import TensorCPEncoding, TensorVMEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.tensorf_field import TensoRFField
from nerfactory.graphs.base import Graph
from nerfactory.graphs.modules.ray_sampler import PDFSampler, UniformSampler
from nerfactory.optimizers.loss import MSELoss
from nerfactory.renderers.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfactory.utils import colors, misc, visualization, writer
from nerfactory.utils.callbacks import Callback


class TensoRFGraph(Graph):
    """
    TensoRF Graph
    """

    def __init__(
        self,
        intrinsics: torch.Tensor = None,
        camera_to_world: torch.Tensor = None,
        enable_density_field: bool = False,
        **kwargs,
    ) -> None:
        self.field = None
        super().__init__(
            intrinsics=intrinsics, camera_to_world=camera_to_world, enable_density_field=enable_density_field, **kwargs
        )

    def get_training_callbacks(self) -> List[Callback]:
        callbacks = []
        if self.density_field is not None:
            callbacks = [
                Callback(
                    update_every_num_iters=self.density_field.update_every_num_iters,
                    func=self.density_field.update_density_grid,
                    density_eval_func=self.field_coarse.density_fn,
                )
            ]
        return callbacks  # type: ignore

    def populate_fields(self):
        """Set the fields."""

        position_encoding = TensorVMEncoding(resolution=8, num_components=1)
        direction_encoding = TensorVMEncoding(resolution=24, num_components=95)

        self.field = TensoRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=2,
            base_mlp_layer_width=128,
        )

    def populate_misc_modules(self):
        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.num_coarse_samples, density_field=self.density_field)
        self.sampler_pdf = PDFSampler(num_samples=self.num_importance_samples, density_field=self.density_field)

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
        param_groups["fields"] = list(self.field.parameters())
        param_groups["position_encoding"] = list(self.field.position_encoding.parameters())
        param_groups["direction_encoding"] = list(self.field.direction_encoding.parameters())
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
