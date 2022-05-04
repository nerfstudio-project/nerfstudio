"""Common Colors"""
from typing import Union
from omegaconf import ListConfig
import torch
from torchtyping import TensorType

WHITE = torch.tensor([1.0, 1.0, 1.0])
BLACK = torch.tensor([0.0, 0.0, 0.0])
RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])

COLORS_DICT = {
    "white": WHITE,
    "black": BLACK,
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
}


def get_color(color: Union[str, list, ListConfig]) -> TensorType[3]:
    """
    Args:
        color (Union[str, list, ListConfig]): Color as a string or a rgb list

    Returns:
        TensorType[3]: Parsed color
    """
    if isinstance(color, str):
        color = color.lower()
        if color not in COLORS_DICT:
            raise ValueError(f"{color} is not a valid preset color")
        return COLORS_DICT[color]
    if isinstance(color, (list, ListConfig)):
        if len(color) != 3:
            raise ValueError(f"Color should be 3 values (RGB) instead got {color}")
        return torch.tensor(color)

    raise ValueError(f"Color should be an RGB list or string, instead got {type(color)}")
