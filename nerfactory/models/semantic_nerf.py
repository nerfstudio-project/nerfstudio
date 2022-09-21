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
Semantic NeRF implementation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from nerfactory.cameras.rays import RayBundle
from nerfactory.datamanagers.structs import Semantics
from nerfactory.fields.modules.encoding import NeRFEncoding
from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.nerf_field import NeRFField
from nerfactory.fields.semantic_nerf_field import SemanticNerfField
from nerfactory.models.base import ModelConfig
from nerfactory.models.modules.scene_colliders import AABBBoxCollider
from nerfactory.models.vanilla_nerf import NeRFModel
from nerfactory.renderers.renderers import SemanticRenderer
from nerfactory.utils import misc


class SemanticNerfModel(NeRFModel):
    """Semantic-NeRF model

    Args:
        config: SemanticNeRF configuration to instantiate model
        semantics: additional semantics data info
    """

    def __init__(self, config: ModelConfig, semantics: Semantics, **kwargs) -> None:
        self.stuff_classes = semantics.stuff_classes
        self.stuff_colors = semantics.stuff_colors
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )
        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

        num_semantic_classes = len(self.stuff_classes)
        self.field_fine = SemanticNerfField(
            num_semantic_classes, position_encoding=position_encoding, direction_encoding=direction_encoding
        )

        # renderers
        self.renderer_semantic = SemanticRenderer()

        # losses
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_bounds=self.scene_bounds)

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
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
        semantic_fine = self.renderer_semantic(field_outputs_fine[FieldHeadNames.SEMANTICS], weights=weights_fine)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "semantic_fine": semantic_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
        semantic_logits = outputs["semantic_fine"]
        semantic_classes = batch["semantics"][..., 0].long()
        semantic_loss_fine = self.cross_entropy_loss(semantic_logits, semantic_classes)
        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "semantic_loss_fine": semantic_loss_fine,
        }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        semantic_logits = outputs["semantic_fine"]
        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantic_logits, dim=-1), dim=-1)  # type: ignore
        semantic_labels_image = self.stuff_colors[semantic_labels]

        mask = batch["mask"].repeat(1, 1, 3)

        images_dict["semantics"] = semantic_labels_image
        images_dict["mask"] = mask
        return metrics_dict, images_dict
