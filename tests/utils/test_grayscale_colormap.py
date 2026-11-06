import pytest
import torch

from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap


@pytest.mark.parametrize("shape", [(5, 1), (2, 4, 1), (2, 3, 4, 1)])
def test_grayscale_preserves_batch_dimensions(shape):
    values = torch.linspace(0, 1, torch.Size(shape).numel()).reshape(shape)
    rgb = apply_colormap(values, ColormapOptions(colormap="gray"))

    assert rgb.shape == (*shape[:-1], 3)
    for channel in range(3):
        torch.testing.assert_close(rgb[..., channel], values[..., 0])
