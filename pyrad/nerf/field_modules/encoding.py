"""
Encoding functions
"""

from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType

from pyrad.nerf.field_modules.base import FieldModule
from pyrad.utils.math import components_from_spherical_harmonics


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

    @abstractmethod
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


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        return self.in_dim

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        return in_tensor


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
        self.b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))

    def get_out_dim(self) -> int:
        return self.num_frequencies * 2

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """
        Args:
            in_tensor (TensorType[..., "input_dim"]): For best performance, the input tensor should be between 0 and 1.

        Returns:
            TensorType[..., "output_dim"]: Output values will be between -1 and 1
        """
        scaled_inputs = in_tensor @ self.b_matrix  # [..., "num_frequencies"]
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
        self.hash_table = nn.Parameter(self.hash_table)

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
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
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


class TensorCPEncoding(Encoding):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF"""

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        """
        Args:
            resolution (int, optional): Resolution of grid. Defaults to 256.
            num_components (int, optional): Number of components per dimension. Defaults to 24.
            init_scale (float, optional): Initialization scale. Defaults to 0.1.
        """
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        in_tensor = in_tensor / 4.0 + 0.5
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution (int): Target resolution.
        """

        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution


class TensorVMEncoding(Encoding):
    """Learned vector-matrix encoding proposed by TensoRF"""

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        """
        Args:
            resolution (int, optional): Resolution of grid. Defaults to 256.
            num_components (int, optional): Number of components per dimension. Defaults to 24.
            init_scale (float, optional): Initialization scale. Defaults to 0.1.
        """
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.plane_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, resolution)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        in_tensor = in_tensor / 4.0 + 0.5
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        plane_features = F.grid_sample(self.plane_coef, plane_coord, align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution (int): Target resolution.
        """

        self.plane_coef.data = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )
        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution


class SHEncoding(Encoding):
    """Spherical harmonic encoding"""

    def __init__(self, levels: int = 4) -> None:
        """
        Args:
            levels (int, optional): Number of spherical hamonic levels to encode. Defaults to 4.
        """
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only suports 1 to 4 levels, requested {levels}")

        self.levels = levels

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def encode(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)
