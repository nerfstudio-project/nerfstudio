# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from typing import Literal

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions


class BGField(Field):
    def __init__(self, appearance_embedding_dim: int, implementation: Literal["tcnn", "torch"] = "torch"):
        super().__init__()
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.mlp_background_color = MLP(
            in_dim=self.direction_encoding.get_out_dim() + appearance_embedding_dim,
            num_layers=2,
            layer_width=64,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_background_rgb(self, ray_bundle: RayBundle, appearance_embedding=None) -> Tensor:
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        if appearance_embedding is not None:
            x = torch.cat([directions_flat, appearance_embedding.repeat(directions_flat.shape[0], 1)], dim=-1)
        else:
            x = directions_flat

        background_rgb = self.mlp_background_color(x).to(directions)

        return background_rgb
