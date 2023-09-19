"""
Regional Nerfacto Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from typing import Dict, Literal, Optional, Tuple
from nerfstudio.field_components.field_heads import FieldHeadNames

try:
    import tinycudann as tcnn
except ImportError:
    pass


class RNerfField(NerfactoField):
    """Regional Nerf Field
    """

    def __init__(
        self, 
        grid_resolutions,
        grid_layers,
        grid_sizes,
        **kwargs) -> None:
        super().__init__(**kwargs)

        self.top_cutoff = -0.5

        self.encs = torch.nn.ModuleList(
            [
                RNerfField._get_encoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.encs])

        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

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
        
        # heights = unnorm_positions[..., 2][..., None]

        # clusters = (density > 0.1).float()
        
        # max_heights = torch.max(heights * clusters, dim=-2).values
        
        # selectors = (heights[..., 0] <= max_heights)
        
        # density = density * selectors[..., None]
        
        return density, base_mlp_out
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = super().get_outputs(ray_samples, density_embedding=density_embedding)

        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.encs]
        x = torch.concat(xs, dim=-1)

        dino_pass = self.dino_net(x).view(*ray_samples.frustums.shape, -1)
        outputs["dino"] = dino_pass

        return outputs
