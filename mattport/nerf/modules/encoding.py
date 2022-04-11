"""
Encoding functions
"""
from base64 import encode
import torch
from torch import nn
from torchtyping import TensorType


class Encoding(nn.Module):
    """Encode an input tensor. Intended to be subclassed"""

    def __init__(self, in_dim: int) -> None:
        """
        Args:
            in_dim (int): Input dimension of tensor
        """
        super().__init__()
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding.

        Returns:
            int: output dimension
        """
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


class ScalingAndOffset(Encoding):
    """Simple scaling and offet to input"""

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        """Each input is scaled then offset

        Args:
            in_dim (int): Input dimension of tensor
            scaling (float, optional): Scaling applied to tensor. Defaults to 1.0.
            offset (float, optional): Offset applied to tensor. Defaults to 0.0.
        """
        super().__init__(in_dim)

        self.in_dim = in_dim
        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        return self.in_dim * self.num_frequencies * 2

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinousoidal encoding proposed in the original NeRF paper"""

    def __init__(self, in_dim: int, num_frequencies: int, min_freq_exp: float, max_freq_exp: float) -> None:
        """Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

        Args:
            in_dim (int): Input dimension of tensor
            num_frequencies (int): Number of encoded frequencies per axis
            min_freq_exp (float): Minimum frequency exponent
            max_freq_exp (float): Maximum frequency exponent
        """
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp

    def get_out_dim(self) -> int:
        return self.in_dim * self.num_frequencies * 2

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """
        Args:
            in_tensor (TensorType[..., "input_dim"]): For best performance, the input tensor should be between 0 and 1.

        Returns:
            TensorType[..., "output_dim"]: Output values will be between -1 and 1
        """
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        scaled_inputs = 2 * torch.pi * in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]
        encoded_inputs = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], axis=-1)
        return encoded_inputs


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
