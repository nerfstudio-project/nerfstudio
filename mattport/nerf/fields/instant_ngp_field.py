"""
Instant-NGP field implementations using tiny-cuda-nn, torch, ....
"""


from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from mattport.nerf.field_modules.encoding import Encoding, HashEncoding, SHEncoding
from mattport.nerf.fields.base import Field
from mattport.nerf.fields.nerf_field import NeRFField
from mattport.structures.rays import PointSamples
from mattport.utils.misc import is_not_none

try:
    import tinycudann as tcnn
except ImportError as e:
    # tinycudann module doesn't exist
    pass


def get_normalized_positions(positions, aabb):
    """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box."""
    aabb_lengths = aabb[1] - aabb[0]
    positions = (positions - aabb[0]) / aabb_lengths
    return positions


class TCNNInstantNGPField(Field):
    """NeRF Field"""

    def __init__(
        self, aabb, num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64
    ) -> None:
        super().__init__()

        # TODO: make this a parameter that is put to the correct device
        self.aabb = Parameter(aabb, requires_grad=False)

        self.geo_feat_dim = geo_feat_dim

        # TODO: set this properly based on the aabb
        per_level_scale = 1.4472692012786865

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_base = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, point_samples: PointSamples):
        """Computes and returns the densities."""
        # TODO: add valid_mask masking!
        # remap positions to range 0 to 1 based on the aabb
        positions = get_normalized_positions(point_samples.positions, self.aabb)
        positions_flat = positions.view(-1, 3)
        dtype = positions_flat.dtype
        x = self.position_encoding(positions_flat)
        h = self.mlp_base(x).view(*point_samples.positions.shape[:-1], -1).to(dtype)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        density = F.softplus(density_before_activation)
        return density, base_mlp_out

    def get_outputs(self, point_samples: PointSamples, density_embedding=None, valid_mask=None):
        # TODO: add valid_mask masking!
        assert is_not_none(density_embedding)
        directions = point_samples.directions
        directions_flat = directions.view(-1, 3)
        dtype = directions_flat.dtype
        d = self.direction_encoding(directions_flat)
        h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        h = self.mlp_head(h).view(*point_samples.directions.shape[:-1], -1).to(dtype)
        rgb = torch.sigmoid(h)
        return {"rgb": rgb}


class TorchInstantNGPField(NeRFField):
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
    ) -> None:
        super().__init__(
            position_encoding,
            direction_encoding,
            base_mlp_num_layers,
            base_mlp_layer_width,
            head_mlp_num_layers,
            head_mlp_layer_width,
            skip_connections,
        )
        self.aabb = Parameter(aabb, requires_grad=False)

    def get_density(self, point_samples: PointSamples):
        normalized_point_samples = point_samples
        normalized_point_samples.positions = get_normalized_positions(normalized_point_samples.positions, self.aabb)
        return super().get_density(normalized_point_samples)


field_implementation_to_class = {"tcnn": TCNNInstantNGPField, "torch": TorchInstantNGPField}
