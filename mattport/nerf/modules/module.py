"""
The module baseclass.
"""
from abc import abstractmethod

from torch import nn
from torchtyping import TensorType


class Module(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self) -> None:
        """Default initialization of module"""
        super().__init__()
        self.in_dim = 0
        self.out_dim = 0

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim (int): input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding.

        Returns:
            int: output dimension
        """
        if not hasattr(self, "out_dim"):
            raise ValueError("Output dimension has not been set")
        if self.out_dim <= 0:
            raise ValueError("Output dimension should be greater than zero")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """_summary_

        Args:
            x (TensorType[..., "input_dim"]): Input tensor to be encoded

        Raises:
            NotImplementedError: _description_

        Returns:
            TensorType[..., "output_dim"]: A encoded input tensor
        """
        raise NotImplementedError
