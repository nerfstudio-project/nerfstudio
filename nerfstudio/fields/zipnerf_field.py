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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from typing import Literal, Optional, Tuple, Callable
from jaxtyping import Float

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.utils.math import erf_approx

EPS = 1.0e-7


class ZipNeRFField(NerfactoField):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
        scale_featurization: scale featurization from appendix of ZipNeRF
        regularize_function: type of regularization
        compute_hash_regularization: whether to compute regularization on hash weights
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        scale_featurization: bool = False,
        regularize_function: Callable[[Tensor], Tensor] = torch.square,
        compute_hash_regularization: bool = True,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(
            aabb=aabb,
            num_images=num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            compute_hash_regularization=compute_hash_regularization,
            regularize_function=regularize_function,
            implementation=implementation,
        )

        self.features_per_level = features_per_level
        self.scale_featurization = scale_featurization

        self.last_dim = self.mlp_base_grid.get_out_dim()

        if scale_featurization:
            self.last_dim += num_levels

            self.mlp_base_mlp = MLP(
                in_dim=self.last_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

    def get_normals(self) -> Float[Tensor, "*batch 3"]:
        """Computes and returns a tensor of normals.

        Args:
            density: Tensor of densities.
        """
        assert self._sample_locations is not None, "Sample locations must be set before calling get_normals."
        assert self._density_before_activation is not None, "Density must be set before calling get_normals."

        normals = torch.autograd.grad(
            self._density_before_activation,
            self._sample_locations,
            grad_outputs=torch.ones_like(self._density_before_activation),
            retain_graph=True,
        )[0]

        # ZipNeRF 6-point hexagonal pattern
        normals = normals.mean(-2)

        normals = -torch.nn.functional.normalize(normals, dim=-1)

        return normals

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities and embeddings."""
        gaussian_samples = ray_samples.frustums.get_multisampled_gaussian_blob(
            rand=self.training,
        )

        assert self.spatial_distortion is not None, "Spatial distortion in ZipNeRF should be defined."

        gaussian_samples = self.spatial_distortion(gaussian_samples)
        mean = gaussian_samples.mean
        cov = gaussian_samples.cov

        # shift mean in cube from [-2, 2] to [0, 1]
        mean = (mean + 2.0) / 4.0
        # cov in range (0, 1)
        cov = cov / 2.0

        self._sample_locations = mean
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        prefix_shape = list(mean.shape[:-1])

        # hash grid performs trilerp inside itself
        mean = (
            self.mlp_base_grid(mean.view(-1, 3))
            .view(prefix_shape + [self.num_levels * self.features_per_level])
            .unflatten(-1, (self.num_levels, self.features_per_level))
        )  # [..., "dim", "num_levels", "features_per_level"]
        weights = erf_approx(
            1 / (8**0.5 * (cov[..., None] * self.mlp_base_grid.scalings.view(-1)).abs()).clamp_min(EPS)
        )  # [..., "dim", "num_levels"]
        features = (
            (mean * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        )  # [..., "dim", "num_levels * features_per_level"]

        if self.scale_featurization:
            with torch.no_grad():
                vl2mean = self.mlp_base_grid.scale_featurization()  # ["num_levels"]
            featurized_weights = (2 * weights.mean(dim=-2) - 1) * (
                self.mlp_base_grid.hash_init_scale**2 + vl2mean
            ).sqrt()  # [..., "num_levels"]
            features = torch.cat([features, featurized_weights], dim=-1)
        features_flat = features.view(-1, self.last_dim)
        h = self.mlp_base_mlp(features_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(mean))
        return density, base_mlp_out
