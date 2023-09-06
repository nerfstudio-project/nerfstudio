from nerfstudio.model_components.renderers import RGBRenderer
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from typing import Generator, Literal, Optional, Tuple, Union
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize

BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None

class RGBRenderer_HDR(RGBRenderer):
    
    def forward(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
        background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.
            background_color: The background color to use for rendering.

        Returns:
            Outputs of rgb values.
        """
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = self.combine_rgb(
            rgb, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        '''annote code below to enable HDR radiance'''
        # if not self.training:
        #     torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb 
