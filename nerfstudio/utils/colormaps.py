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

""" Helper functions for visualizing outputs """

from typing import Optional

import torch
from matplotlib import cm
from torchtyping import TensorType

from nerfstudio.utils import colors


def apply_colormap(image: TensorType["bs":..., 1], cmap="viridis") -> TensorType["bs":..., "rgb":3]:
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        cmap: Colormap for image.

    Returns:
        TensorType: Colored image with colors in [0, 1]
    """

    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image = torch.nan_to_num(image, 0)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


def apply_depth_colormap(
    depth: TensorType["bs":..., 1],
    accumulation: Optional[TensorType["bs":..., 1]] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    cmap="turbo",
) -> TensorType["bs":..., "rgb":3]:
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        cmap: Colormap to apply.

    Returns:
        Colored depth image with colors in [0, 1]
    """

    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_boolean_colormap(
    image: TensorType["bs":..., 1, bool],
    true_color: TensorType["bs":..., "rgb":3] = colors.WHITE,
    false_color: TensorType["bs":..., "rgb":3] = colors.BLACK,
) -> TensorType["bs":..., "rgb":3]:
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image
