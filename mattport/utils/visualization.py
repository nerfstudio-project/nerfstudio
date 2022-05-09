""" Helper functions for visualizing outputs """

from typing import Optional
import torch
from matplotlib import cm
from torchtyping import TensorType


def apply_colormap(image: TensorType[..., 1], cmap="viridis") -> TensorType[..., 3]:
    """Convert single channel to a color image.

    Args:
        image (TensorType[..., 1]): Single channel image.
        cmap (str, optional): Colormap for image. Defaults to 'turbo'.

    Returns:
        TensorType[..., 3]: Colored image
    """

    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)

    image2 = (image * 255).long()

    if torch.min(image2) < 0 or torch.max(image2) > 255:
        print("hi")

    image = (image * 255).long()

    return colormap[image[..., 0]]


def apply_depth_colormap(
    depth: TensorType[..., 1],
    accumulation: Optional[TensorType[..., 1]] = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    cmap="turbo",
) -> TensorType[..., 3]:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (TensorType[..., 1]): Depth image.
        accumulation (TensorType[..., 1], optional): Ray accumulation used for masking vis. Defaults to None.
        near_plane (float, optional): Closest depth to consider. If None, use min image value. Defaults to None.
        far_plane (float, optional): Furthest depth to consider. If None, use max image value. Defaults to None.
        cmap (str, optional): Colormap to apply. Defaults to "turbo".

    Returns:
        TensorType[..., 3]: Colored depth image
    """

    near_plane = near_plane or torch.min(depth)
    far_plane = far_plane or torch.max(depth)

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image
