""" Helper functions for visualizing outputs """

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

    image = (image * 255).long()

    return colormap[image]
