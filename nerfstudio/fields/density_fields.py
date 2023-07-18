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

"""
Proposal network field.
"""


from typing import Literal, Optional, Tuple, Callable

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldHeadNames
from nerfstudio.utils.math import erf_approx

EPS = 1.0e-7


class HashMLPDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        regularize_function: Callable[[Tensor], Tensor] = torch.square,
        compute_hash_regularization: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        self.regularize_function = regularize_function
        self.compute_hash_regularization = compute_hash_regularization

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        if not self.use_linear:
            self.network = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_base = torch.nn.Sequential(self.encoding, self.network)
        else:
            self.network = torch.nn.Linear(self.encoding.get_out_dim(), 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.network(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: Optional[RaySamples] = None, density_embedding: Optional[Tensor] = None) -> dict:
        outputs = {}
        if self.compute_hash_regularization:
            outputs[FieldHeadNames.HASH_DECAY] = self.encoding.regularize_hash_pyramid(self.regularize_function)

        return outputs


class HashMLPGaussianDensityField(HashMLPDensityField):
    """A lightweight density field module for Gaussians.
    Implemented from ZipNeRF paper ideas.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        scale_featurization: bool = False,
        regularize_function: Callable[[Tensor], Tensor] = torch.square,
        compute_hash_regularization: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:

        super().__init__(
            aabb=aabb,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            spatial_distortion=spatial_distortion,
            use_linear=use_linear,
            num_levels=num_levels,
            max_res=max_res,
            base_res=base_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            regularize_function=regularize_function,
            compute_hash_regularization=compute_hash_regularization,
            implementation=implementation,
        )

        self.features_per_level = features_per_level
        self.scale_featurization = scale_featurization

        self.last_dim = self.encoding.get_out_dim()

        if scale_featurization:
            self.last_dim += num_levels

            if not self.use_linear:
                self.network = MLP(
                    in_dim=self.last_dim,
                    num_layers=num_layers,
                    layer_width=hidden_dim,
                    out_dim=1,
                    activation=nn.ReLU(),
                    out_activation=None,
                    implementation=implementation,
                )
                self.mlp_base = torch.nn.Sequential(self.encoding, self.network)
            else:
                self.network = torch.nn.Linear(self.last_dim, 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        """Computes and returns the densities."""
        gaussian_samples = ray_samples.frustums.get_multisampled_gaussian_blob(rand=self.training)

        assert self.spatial_distortion is not None, "Spatial distortion in ZipNeRF should be defined."

        gaussian_samples = self.spatial_distortion(gaussian_samples)
        mean = gaussian_samples.mean
        cov = gaussian_samples.cov

        # shift mean in cube from [-2, 2] to [0, 1]
        mean = (mean + 2.0) / 4.0
        # cov in range [0, 1]
        cov = cov / 2.0

        prefix_shape = list(mean.shape[:-1])

        # hash grid performs trilerp inside itself
        mean = self.encoding(mean.view(-1, 3)).view(
            prefix_shape + [self.num_levels * self.features_per_level]
        ).unflatten(-1, (self.num_levels, self.features_per_level)) # [..., "dim", "num_levels", "features_per_level"]
        weights = erf_approx(1 / (8 ** 0.5 * cov[..., None] * self.encoding.scalings.view(-1)).abs().clamp_min(EPS)) # [..., "dim", "num_levels"]

        features = (mean * weights[..., None]).mean(dim=-3).flatten(-2, -1) # [..., "dim", "num_levels * features_per_level"]

        if self.scale_featurization:
            with torch.no_grad():
                vl2mean = self.encoding.scale_featurization() # ["num_levels"]
            featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoding.hash_init_scale ** 2 + vl2mean).sqrt() # [..., "num_levels"]
            features = torch.cat([features, featurized_w], dim=-1)
        features_flat = features.view(-1, self.last_dim)
        density_before_activation = self.network(features_flat).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(mean))
        return density, None
