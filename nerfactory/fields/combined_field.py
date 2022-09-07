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

"""


# from typing import Optional, Tuple

from typing import Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from nerfactory.cameras.rays import RaySamples
from nerfactory.datamanagers.structs import SceneBounds

# from nerfactory.fields.base import Field
from nerfactory.fields.instant_ngp_field import TCNNInstantNGPField
from nerfactory.fields.modules.encoding import Encoding, HashEncoding, SHEncoding

# from nerfactory.fields.modules.encoding import Encoding, HashEncoding, SHEncoding
# from nerfactory.fields.modules.field_heads import FieldHeadNames
from nerfactory.fields.modules.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfactory.fields.nerf_field import NeRFField
from nerfactory.utils.activations import trunc_exp

# from torch.nn.parameter import Parameter

# from torchtyping import TensorType


try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]"""
    return (directions + 1.0) / 2.0


class TCNNCombinedField(TCNNInstantNGPField):
    """NeRF Field"""

    def __init__(
        self,
        aabb,
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__(self, aabb)
        self.spatial_distortion = spatial_distortion

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = self.spatial_distortion(ray_samples.frustums.get_positions())
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out


class TorchCombinedField(NeRFField):
    """
    PyTorch implementation of the instant-ngp field.
    """

    def __init__(
        self,
        aabb,
        position_encoding: Encoding = HashEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 3,
        base_mlp_layer_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        skip_connections: Tuple = (4,),
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__(
            position_encoding,
            direction_encoding,
            base_mlp_num_layers,
            base_mlp_layer_width,
            head_mlp_num_layers,
            head_mlp_layer_width,
            skip_connections,
            spatial_distortion=spatial_distortion,
        )
        self.aabb = Parameter(aabb, requires_grad=False)


field_implementation_to_class = {"tcnn": TCNNCombinedField, "torch": TorchCombinedField}
