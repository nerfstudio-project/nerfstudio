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

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType
from pyrad.data.structs import Semantics

from pyrad.fields.modules.encoding import Encoding, Identity, NeRFEncoding
from pyrad.fields.modules.field_heads import DensityFieldHead, FieldHeadNames, RGBFieldHead, SemanticFieldHead
from pyrad.fields.modules.mlp import MLP
from pyrad.fields.base import Field
from pyrad.fields.nerf_field import NeRFField
from pyrad.graphs.vanilla_nerf import NeRFGraph
from pyrad.renderers.renderers import SemanticRenderer
from pyrad.cameras.rays import RaySamples, RayBundle
from pyrad.utils import writer


class SemanticNerfField(Field):
    """Semantic-NeRF field"""

    def __init__(
        self,
        num_semantic_classes: int,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
    ) -> None:
        super().__init__()
        self.num_semantic_classes = num_semantic_classes
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
        )
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
        )
        self.mlp_semantic = MLP(
            in_dim=self.mlp_head.get_out_dim(),
            layer_width=self.mlp_head.layer_width // 2,
            num_layers=1,
            activation=nn.ReLU(),
        )
        self.field_head_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_head_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())
        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(), num_classes=self.num_semantic_classes
        )

    def get_density(self, ray_samples: RaySamples):
        encoded_xyz = self.position_encoding(ray_samples.frustums.get_positions())
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_head_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))
        outputs = {}
        # rgb
        outputs[self.field_head_rgb.field_head_name] = self.field_head_rgb(mlp_out)
        # semantic
        mlp_out_sem = self.mlp_semantic(mlp_out)
        outputs[self.field_head_semantic.field_head_name] = self.field_head_semantic(mlp_out_sem)
        return outputs


class SemanticNerfGraph(NeRFGraph):
    """Semantic-NeRF graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, semantics: Semantics = None, **kwargs) -> None:
        self.stuff_classes = semantics.stuff_classes
        self.stuff_colors = semantics.stuff_colors
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

        num_semantic_classes = len(self.stuff_classes)
        self.field_fine = SemanticNerfField(
            num_semantic_classes, position_encoding=position_encoding, direction_encoding=direction_encoding
        )

    def populate_misc_modules(self):
        super().populate_misc_modules()
        self.renderer_semantic = SemanticRenderer()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")

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

    def get_loss_dict(self, outputs, batch):
        image = batch["image"]
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
        return loss_dict

    def log_test_image_outputs(self, image_idx, step, batch, outputs):
        super().log_test_image_outputs(image_idx, step, batch, outputs)
        semantic_logits = outputs["semantic_fine"]
        semantic_labels = torch.argmax(torch.nn.functional.softmax(semantic_logits, dim=-1), dim=-1)
        semantic_labels_image = self.stuff_colors[semantic_labels]
        writer.put_image(name=f"semantics/image_idx_{image_idx}", image=semantic_labels_image, step=step)

        mask = batch["mask"].repeat(1, 1, 3)
        writer.put_image(name=f"mask/image_idx_{image_idx}", image=mask, step=step)
