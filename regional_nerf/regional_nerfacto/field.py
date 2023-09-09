"""
Regional Nerfacto Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

import torch
from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from typing import Dict, Literal, Optional, Tuple


class RNerfField(NerfactoField):
    """Regional Nerf Field
    """

    def __init__(
        self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.top_cutoff = -0.5

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(unnorm_positions)
            positions = (positions + 2.0) / 4.0
        else:
            unnorm_positions = ray_samples.frustums.get_positions()
            positions = SceneBox.get_normalized_positions(unnorm_positions, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        # Selector to mask out positions with z higher than -0.5
        selector_0 = (unnorm_positions[..., 2] <= self.top_cutoff) # Navlab added
            
        selector = selector_0 & ((positions > 0.0) & (positions < 1.0)).all(dim=-1)

        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out