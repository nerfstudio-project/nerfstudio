"""
Encoding functions
"""
from torch import nn
from torchtyping import TensorType


class Encoding(nn.Module):
    """Encode an input tensor. Intended to be subclassed"""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim

        raise NotImplementedError

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Encodes an input tensor.

        Args:
            x (TensorType[..., "input_dim"]): Input tensor to be encoded

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
        """
        raise NotImplementedError

    def forward(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Call forward"""
        return self.encode(in_tensor)


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding"""

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Encodes an input tensor.

        Args:
            x (TensorType[..., "input_dim"]): Input tensor to be encoded

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
        """
        raise NotImplementedError


class HashEncoding(Encoding):
    """Hash encoding"""

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Encodes an input tensor.

        Args:
            x (TensorType[..., "input_dim"]): Input tensor to be encoded

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
        """
        raise NotImplementedError


encodings_dict = {
    "RFF": RFFEncoding,
    "hash": HashEncoding,
}
