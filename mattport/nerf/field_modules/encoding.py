"""
Encoding functions
"""
import numpy as np
import torch
from torchtyping import TensorType
from mattport.nerf.field_modules.base import FieldModule


class Encoding(FieldModule):
    """Encode an input tensor. Intended to be subclassed"""

    def __init__(self, in_dim: int) -> None:
        """
        Args:
            in_dim (int): Input dimension of tensor
        """
        super().__init__()
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

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

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        return self.in_dim

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinousoidal encoding proposed in the original NeRF paper"""

    def __init__(
        self, in_dim: int, num_frequencies: int, min_freq_exp: float, max_freq_exp: float, include_input: bool = False
    ) -> None:
        """Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

        Args:
            in_dim (int): Input dimension of tensor
            num_frequencies (int): Number of encoded frequencies per axis
            min_freq_exp (float): Minimum frequency exponent
            max_freq_exp (float): Maximum frequency exponent
            include_input (float): Append the input coordinate to the encoding
        """
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """
        Args:
            in_tensor (TensorType[..., "input_dim"]): For best performance, the input tensor should be between 0 and 1.

        Returns:
            TensorType[..., "output_dim"]: Output values will be between -1 and 1
        """
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = 2 * torch.pi * in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]
        encoded_inputs = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], axis=-1)
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], axis=-1)
        return encoded_inputs


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding"""

    def __init__(self, in_dim: int, num_frequencies: int, scale: float) -> None:
        """

        Args:
            in_dim (int): Input dimension of tensor
            num_frequencies (int): Number of encoding frequencies
            scale (float): Std of Gaussian to sample frequencies. Must be greater than zero
        """
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        if not scale > 0:
            raise ValueError("RFF encoding scale should be greater than zero")
        self.scale = scale

    def get_out_dim(self) -> int:
        return self.num_frequencies * 2

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """
        Args:
            in_tensor (TensorType[..., "input_dim"]): For best performance, the input tensor should be between 0 and 1.

        Returns:
            TensorType[..., "output_dim"]: Output values will be between -1 and 1
        """
        b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))
        scaled_inputs = in_tensor @ b_matrix  # [..., "num_frequencies"]
        encoded_inputs = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], axis=-1)
        return encoded_inputs


class HashEncoding(Encoding):
    """Hash encoding"""

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        hash_table_size: int = 2**19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
    ) -> None:
        """_summary_

        Args:
            num_levels (int, optional): Number of feature grids. Defaults to 16.
            min_res (int, optional): Resolution of smallest feature grid. Defaults to 16.
            max_res (int, optional): Resolution of largest feature grid. Defaults to 1024.
            hash_table_size (int, optional): Size of hash table. Defaults to 2**19.
            features_per_level (int, optional): Number of features per level. Defaults to 2.
            hash_init_scale (float, optional): Value to initialize hash grid. Defaults to 0.001.
        """
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.hash_table_size = hash_table_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * hash_table_size
        self.hash_table = torch.rand(size=(hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = torch.nn.Parameter(self.hash_table)

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: TensorType[..., "num_levels", 3]) -> TensorType[..., "num_levels"]:
        """Hash tensor using method described in Instant-NGP

        Args:
            in_tensor (TensorType[..., &quot;num_levels&quot;, 3]): Tensor to be hashed

        Returns:
            TensorType[..., &quot;num_levels&quot;]: Hashed tensor
        """
        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        assert in_tensor.shape[-1] == 3

        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(1, 1, -1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]
