"""
The Graph module contains all trainable parameters.
"""
from dataclasses import dataclass
from torch import nn


@dataclass
class GraphInputs:
    """Datastucture to encode the inputs to the graph."""

    points: float


class Graph(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self):
        """_summary_"""
        raise NotImplementedError
