from typing import Literal

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components import MLP
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions


class SplatfactoWField(Field):
    def __init__(
        self,
        appearance_embed_dim,
        appearance_features_dim,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        super().__init__()
        self.color_nn = MLP(
            in_dim=appearance_embed_dim + appearance_features_dim,
            num_layers=2,
            layer_width=256,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def forward(self, appearance_embed: Tensor, appearance_features: Tensor):
        color_out = self.color_nn(torch.cat((appearance_embed, appearance_features), dim=-1))
        return color_out
