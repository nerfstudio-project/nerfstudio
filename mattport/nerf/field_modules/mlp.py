"""
Multi Layer Perceptron
"""
from typing import Optional

from torch import nn
from torchtyping import TensorType
from mattport.nerf.field_modules.base import FieldModule


class MLP(FieldModule):
    """Multilayer perceptron"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        layer_width: int,
        activation: Optional[nn.Module] = None,
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        """Initialize parameters of mulilayer perceptron.

        Args:
            in_dim (int): Input layer dimension
            out_dim (int): Ouput layer dimension
            num_layers (int): Number of network layers
            layer_width (int): Width of each MLP layer
            activation (Optional[nn.Module], optional): intermediate layer activation function. Defaults to nn.ReLU.
            out_activation (Optional[nn.Module], optional): output activation function. Defaults to None.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

    def build_nn_modules(self) -> None:
        """Initialize mulilayer perceptron."""
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.in_dim, self.layer_width))
            else:
                layers.append(nn.Linear(self.layer_width, self.layer_width))
            if self.activation is not None:
                layers.append(self.activation)
        layers.append(nn.Linear(self.layer_width, self.out_dim))
        if self.out_activation is not None:
            layers.append(self.out_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, in_tensor: TensorType[..., "in_dim"]) -> TensorType[..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor (TensorType[..., "in_dim]): Network input

        Returns:
            TensorType[..., "out_dim"]: Network output
        """

        return self.net(in_tensor)
