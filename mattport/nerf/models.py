"""
Collection of Models
"""
from typing import Optional

from torch import nn
from torchtyping import TensorType


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        layer_width: int,
        activation: Optional[nn.Module] = nn.ReLU,
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        """Initialize mulilayer perceptron.

        Args:
            in_dim (int): Input layer dimension
            out_dim (int): Ouput layer dimension
            num_layers (int): Number of network layers
            layer_width (int): Width of each MLP layer
            activation (Optional[nn.Module], optional): intermediate layer activation function. Defaults to nn.ReLU.
            out_activation (Optional[nn.Module], optional): output activation function. Defaults to None.
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, layer_width))
            else:
                layers.append(nn.Linear(layer_width, layer_width))
            if activation is not None:
                layers.append(activation)
        layers.append(nn.Linear(layer_width, out_dim))
        if out_activation is not None:
            layers.append(out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> TensorType[..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Network output
        """

        return self.net(in_tensor)
