# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Proposal network field.
"""


from typing import Optional, Dict

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class HashMLPDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
        self,
        aabb,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear=False,
        num_levels=8,
        max_res=1024,
        base_res=16,
        log2_hashmap_size=18,
        features_per_level=2,
        wavelength_embedding: Optional[tcnn.Encoding]=None,
        latent_embedding_dim: int=7,
        wavelength_layers: int=2,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        self.wavelength_embedding = wavelength_embedding
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        if not self.use_linear:
            if wavelength_embedding is not None:
                config["network"]["output_activation"] = "ReLU"
                self.mlp_base = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3,
                    n_output_dims=latent_embedding_dim,
                    encoding_config=config["encoding"],
                    network_config=config["network"],
                )
                self.mlp_wavelength = tcnn.Network(n_input_dims=latent_embedding_dim + wavelength_embedding.n_output_dims,
                                                    n_output_dims=1,
                                                    network_config={
                                                        "otype": "FullyFusedMLP",
                                                        "activation": "ReLU",
                                                        "output_activation": "None",
                                                        "n_neurons": hidden_dim,
                                                        "n_hidden_layers": wavelength_layers - 1,
                                                    })
            else:
                self.mlp_base = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3,
                    n_output_dims=1,
                    encoding_config=config["encoding"],
                    network_config=config["network"],
                )
        else:
            self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config["encoding"])
            self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)

    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)
        
        if self.wavelength_embedding is not None:
            wavelengths = ray_samples.metadata["wavelengths"]
            wavelengths_flat = wavelengths.reshape(-1, 1)
            wavelengths_embedding = self.wavelength_embedding(wavelengths_flat).view(*ray_samples.shape, -1)
            x = torch.cat([density_before_activation, wavelengths_embedding], dim=-1)
            density_before_activation = self.mlp_wavelength(x.view(-1, self.mlp_wavelength.n_input_dims)).view(*ray_samples.frustums.shape, -1).to(positions)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}

    def density_fn(self, positions: TensorType["bs":..., 3], metadata: Optional[Dict[str, TensorType["bs":..., 1]]]=None) -> TensorType["bs":..., 1]:
        """Returns only the density. Used primarily with the density grid.

        Args:
            positions: the origin of the samples/frustums
        """
        # Need to figure out a better way to describe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            metadata=metadata if self.wavelength_embedding is not None else None,
        )
        density, _ = self.get_density(ray_samples)
        return density
